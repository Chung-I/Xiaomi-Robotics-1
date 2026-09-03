# Copyright (C) 2026 Xiaomi Corporation.
"""Pure analysis dataset for the XR1 mass/CoM Plan-2 probing study, Task 1.

Frozen interfaces (consumed by Plan-2 Tasks 2 and 4 -- do not change the
returned dict's keys or ``window_stack``/``phase_masks``'s signatures
without updating those tasks):

- :func:`phase_masks` -- event-derived (never proxy-derived, per the
  study's Global Constraints) per-step phase masks: ``precontact`` (before
  the first grasped step), ``grasp`` (first-grasp .. liftoff, exclusive of
  the liftoff step itself -- that step is the first ``carry`` step),
  ``carry`` (airborne: ``liftoff_step`` up to, but excluding, the first
  step after liftoff where the object's z drops back below
  ``init_z + carry_return_margin_m``, or the episode end if it never
  does), ``all`` (every step). A never-grasped episode is all
  ``precontact``; a grasped-but-never-lifted episode (``liftoff_step``
  ``-1``, ``recorder.liftoff_step``'s convention) has ``grasp`` run to the
  episode end and an EMPTY ``carry`` -- Task 1 Step 1's fixture.
- :func:`mass_log_c` -- ``log(mass_kg / 0.6)`` with intended-mass snapping:
  the npz's measured ``mass_kg`` scalar is fp-noisy (it is read live off
  ``env.sim.model.body_mass``, not the literal ``conditions.MASS_LEVELS_KG``
  constant), so this snaps to the nearest of ``{0.15, 0.6, 1.2}`` before
  taking the log, and raises if the snap residual is not tiny (``atol``
  ``1e-6`` default) -- a real deviation that large would mean the npz's
  condition/mass are inconsistent, which must be loud, not silently
  averaged away.
- :func:`window_stack` -- pure left-clamped history window, ``(T, ...) ->
  (T, k, ...)``, matching ``eval_robocasa365.entry.sample_history``'s index
  formula exactly (see that function's docstring) so the certificate/probe
  code building the XR1 4-frame proprio window at analysis time reproduces
  precisely what the live policy loop saw, without needing a live deque.
- :func:`deficit_z` -- ``commanded_delta[..., 2] - achieved_eef_delta[..., 2]``
  per step (the plan's exact formula; both channels come straight off the
  Phase-1 npz's own ``commanded_delta``/``achieved_eef_delta`` -- see
  ``recorder.py``'s frame-note docstring for the base-vs-world-frame
  caveat those two channels carry).
- :func:`load_study` -- joins one model's 105 Phase-1 npz (ground truth:
  force/tracking/mass/mask channels) with the matching 105 Task-1
  ``extract_policy_state.py`` output npz (``obs_state_14``/``obs_state_16``)
  into ONE concatenated per-step dict, by ``(condition, seed)`` -- the two
  corpora are asserted to have identical per-episode step counts (a T
  mismatch means the replay that produced the policy-state npz diverged
  from the recording, which Task 1 Step 2's bit-exactness assert should
  already have caught at extraction time; this is a second, cheap guard at
  load time).

No sim imports anywhere in this module -- only numpy (``load_study`` reads
plain npz files via ``numpy.load``, never touches ``robocasa``/``gym``), so
it imports fast and works without the robocasa venv's sim extras.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Duplicated (not imported) from entry_mass.py / run_phase1.py: importing
# either at module level pulls in progressively heavier chains (entry_mass
# -> eval_robocasa365.entry -> deploy/client.py's torch/transformers stack;
# run_phase1 -> subprocess/signal/socket server-lifecycle code) that this
# pure analysis module has no business depending on -- see this module's
# docstring's "No sim imports anywhere" note and recorder.py's precedent
# for the same call (its own module docstring).
PRIMARY_ENV = "PickPlaceCounterToCabinet"
MASS_CONDITIONS = ("MassLight", "MassMedium", "MassHeavy")
DEFAULT_BASE_SEED = 7
DEFAULT_N_SEEDS = 35

CARRY_RETURN_MARGIN_M = 0.02

MASS_LEVELS_KG = (0.15, 0.6, 1.2)
MASS_LOG_REF_KG = 0.6
MASS_SNAP_ATOL_KG = 1e-6


# ---------------------------------------------------------------------------
# phase_masks (pure)
# ---------------------------------------------------------------------------


def phase_masks(
    grasped: Any,
    liftoff_step: int,
    obj_z: Any,
    init_z: float,
    carry_return_margin_m: float = CARRY_RETURN_MARGIN_M,
) -> dict[str, np.ndarray]:
    """Event-derived per-step phase masks for one episode.

    ``grasped`` (T,) bool, ``liftoff_step`` (the npz scalar; ``-1`` means
    never lifted, per ``recorder.liftoff_step``'s convention), ``obj_z``
    (T,) the object's world z each step, ``init_z`` the object's step-0
    height (its resting height before contact -- pass ``obj_z[0]``, not a
    recomputed baseline, so this stays event-derived).

    Returns ``{"precontact", "grasp", "carry", "all"}``, each a (T,) bool
    array. The three phase masks are NOT guaranteed to partition every
    step (e.g. the steps after a carry return-to-height belong to none of
    the three) -- ``all`` is the catch-all for whole-episode baselines.
    """
    grasped_arr = np.asarray(grasped, dtype=bool).reshape(-1)
    obj_z_arr = np.asarray(obj_z, dtype=float).reshape(-1)
    steps = grasped_arr.shape[0]
    if obj_z_arr.shape[0] != steps:
        raise ValueError(
            f"phase_masks: grasped and obj_z must have the same length, got "
            f"{steps} and {obj_z_arr.shape[0]}"
        )

    precontact = np.zeros(steps, dtype=bool)
    grasp = np.zeros(steps, dtype=bool)
    carry = np.zeros(steps, dtype=bool)
    all_mask = np.ones(steps, dtype=bool)

    grasped_idx = np.flatnonzero(grasped_arr)
    if grasped_idx.size == 0:
        # Never grasped: every step is "before the first grasped step".
        precontact[:] = True
        return {"precontact": precontact, "grasp": grasp, "carry": carry, "all": all_mask}

    first_grasp = int(grasped_idx[0])
    precontact[:first_grasp] = True

    lift = int(liftoff_step)
    if lift < 0:
        # Grasped but never lifted 5 cm (recorder.liftoff_step's rise
        # threshold): grasp runs to the episode end; carry stays empty --
        # Task 1 Step 1's "no-lift episode -> carry empty" fixture.
        grasp[first_grasp:] = True
        return {"precontact": precontact, "grasp": grasp, "carry": carry, "all": all_mask}

    if lift >= steps:
        raise ValueError(f"phase_masks: liftoff_step {lift} is out of range for {steps} steps")

    # grasp ends (exclusive) at liftoff -- the liftoff step itself is the
    # FIRST carry step, not the last grasp step.
    grasp[first_grasp:lift] = True

    threshold = init_z + carry_return_margin_m
    after_liftoff = obj_z_arr[lift + 1 :] < threshold
    if after_liftoff.any():
        # argmax of a bool array is the first True -- the z-return step,
        # which becomes carry's exclusive end (boundary exact: that step
        # itself is NOT carry).
        carry_end = lift + 1 + int(np.argmax(after_liftoff))
    else:
        carry_end = steps
    carry[lift:carry_end] = True

    return {"precontact": precontact, "grasp": grasp, "carry": carry, "all": all_mask}


# ---------------------------------------------------------------------------
# mass_log_c (pure)
# ---------------------------------------------------------------------------


def mass_log_c(
    mass_kg: Any,
    levels_kg: tuple[float, ...] = MASS_LEVELS_KG,
    ref_kg: float = MASS_LOG_REF_KG,
    atol: float = MASS_SNAP_ATOL_KG,
) -> Any:
    """``log(mass_kg / ref_kg)`` after snapping ``mass_kg`` to the nearest
    of ``levels_kg`` -- see the module docstring. Accepts a scalar or an
    array; returns the same shape (a Python float for a scalar input).

    Raises ``ValueError`` if any value's snap residual is >= ``atol`` --
    per the plan, this must never be loosened to paper over a genuine
    condition/mass mismatch.
    """
    values = np.asarray(mass_kg, dtype=float)
    scalar_input = values.ndim == 0
    flat = values.reshape(-1)
    levels = np.asarray(levels_kg, dtype=float)

    diffs = np.abs(flat[:, None] - levels[None, :])
    nearest_idx = np.argmin(diffs, axis=1)
    snapped = levels[nearest_idx]
    residual = np.abs(flat - snapped)

    bad = residual >= atol
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"mass_log_c: measured mass {flat[i]!r} kg is {residual[i]:.3e} kg away "
            f"from the nearest intended level {snapped[i]!r} kg (atol {atol}) -- not "
            f"an fp-noise-only deviation from {levels_kg}"
        )

    result = np.log(snapped / ref_kg).reshape(values.shape)
    return float(result) if scalar_input else result


# ---------------------------------------------------------------------------
# window_stack (pure)
# ---------------------------------------------------------------------------


def window_stack(x: Any, k: int = 4, stride: int = 2) -> np.ndarray:
    """Left-clamped history window: ``(T, ...) -> (T, k, ...)``.

    For each output step ``t``, window position ``i`` (``0..k-1``, newest
    last) is ``x[max(0, t - (k - 1 - i) * stride)]`` -- exactly
    ``eval_robocasa365.entry.sample_history``'s index formula (there,
    ``len(items) - 1`` plays the role of ``t``), just applied to a full
    array instead of a live bounded deque. Early steps repeat ``x[0]``
    (left-edge clamp) rather than raising or padding with zeros.
    """
    x = np.asarray(x)
    if x.ndim == 0:
        raise ValueError("window_stack: x must have at least one dimension (a time axis)")
    if k < 1 or stride < 1:
        raise ValueError(f"window_stack: k and stride must be >= 1, got k={k} stride={stride}")

    steps = x.shape[0]
    offsets = np.array([(k - 1 - i) * stride for i in range(k)], dtype=np.int64)  # (k,)
    t = np.arange(steps, dtype=np.int64)[:, None]  # (T, 1)
    idx = np.maximum(0, t - offsets[None, :])  # (T, k)
    return x[idx]


# ---------------------------------------------------------------------------
# deficit_z (pure)
# ---------------------------------------------------------------------------


def deficit_z(commanded_delta: Any, achieved_eef_delta: Any) -> np.ndarray:
    """``commanded_delta[..., 2] - achieved_eef_delta[..., 2]`` -- the
    plan's exact formula (commanded z-motion minus achieved z-motion, in
    that order; both channels are per-step (T, 6) arrays off the Phase-1
    npz -- see ``recorder.py``'s frame-note docstring for the base- vs
    world-frame caveat on these two channels' axis-wise decompositions).
    """
    commanded = np.asarray(commanded_delta, dtype=np.float64)
    achieved = np.asarray(achieved_eef_delta, dtype=np.float64)
    if commanded.shape != achieved.shape:
        raise ValueError(
            f"deficit_z: commanded_delta {commanded.shape} and achieved_eef_delta "
            f"{achieved.shape} must have the same shape"
        )
    return (commanded[..., 2] - achieved[..., 2]).astype(np.float32)


# ---------------------------------------------------------------------------
# load_study
# ---------------------------------------------------------------------------


def _phase1_npz_path(phase1_root: Path, cell: str, model: str, condition: str, seed: int) -> Path:
    """``run_phase1.cell_dir_for``'s convention, duplicated (not imported --
    see the module docstring): XR1 keeps the unsuffixed cell dir; every
    other model gets a ``__<model>`` suffix.
    """
    cell_dir = cell if model == "xr1" else f"{cell}__{model}"
    return phase1_root / cell_dir / condition / f"ep_{seed}.npz"


def _policy_state_npz_path(policy_state_root: Path, model: str, condition: str, seed: int) -> Path:
    return policy_state_root / model / condition / f"ep_{seed}.npz"


def load_study(
    model: str,
    phase1_root: str | Path,
    policy_state_root: str | Path,
    cell: str = PRIMARY_ENV,
    conditions: Iterable[str] = MASS_CONDITIONS,
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Concatenated per-step arrays for one model, joining the Phase-1
    ground-truth npz with Task 1's policy-state npz over every
    ``(condition, seed)`` pair.

    ``seeds`` defaults to the study's 35 matched-pair seeds
    (``conditions.episode_seeds(7, 0, 35)`` == ``range(7, 42)``); pass an
    explicit subset for tests/spot checks.

    Returns a dict with keys ``model, episode_id, seed, condition, step,
    mass_kg, mass_log_c, masks (dict of precontact/grasp/carry/all),
    deficit_z, ee_force, ee_torque, cfrc_obj, commanded_delta,
    achieved_eef_delta, grasped, obj_pos, policy_state_14, policy_state_16``
    -- every per-step array is (N,) or (N, D) over ALL episodes
    concatenated in ``(condition, seed)`` iteration order (conditions in
    the order given, seeds ascending within each).
    """
    phase1_root = Path(phase1_root)
    policy_state_root = Path(policy_state_root)
    conditions = tuple(conditions)
    if seeds is None:
        seed_list = list(range(DEFAULT_BASE_SEED, DEFAULT_BASE_SEED + DEFAULT_N_SEEDS))
    else:
        seed_list = list(seeds)

    episode_id: list[str] = []
    condition_col: list[str] = []
    seed_chunks: list[np.ndarray] = []
    step_chunks: list[np.ndarray] = []
    mass_kg_chunks: list[np.ndarray] = []
    deficit_chunks: list[np.ndarray] = []
    ee_force_chunks: list[np.ndarray] = []
    ee_torque_chunks: list[np.ndarray] = []
    cfrc_obj_chunks: list[np.ndarray] = []
    commanded_delta_chunks: list[np.ndarray] = []
    achieved_eef_delta_chunks: list[np.ndarray] = []
    grasped_chunks: list[np.ndarray] = []
    obj_pos_chunks: list[np.ndarray] = []
    policy_state_14_chunks: list[np.ndarray] = []
    policy_state_16_chunks: list[np.ndarray] = []
    mask_chunks: dict[str, list[np.ndarray]] = {
        "precontact": [], "grasp": [], "carry": [], "all": [],
    }

    for condition in conditions:
        for seed in seed_list:
            p1_path = _phase1_npz_path(phase1_root, cell, model, condition, seed)
            ps_path = _policy_state_npz_path(policy_state_root, model, condition, seed)

            with np.load(p1_path) as data:
                p1 = {key: data[key] for key in data.files}
            with np.load(ps_path) as data:
                ps = {key: data[key] for key in data.files}

            grasped = np.asarray(p1["grasped"], dtype=bool).reshape(-1)
            steps = grasped.shape[0]
            obj_pos = np.asarray(p1["obj_pos"], dtype=np.float32).reshape(steps, 3)
            liftoff = int(np.asarray(p1["liftoff_step"]))
            masks = phase_masks(grasped, liftoff, obj_pos[:, 2], float(obj_pos[0, 2]))

            state14 = np.asarray(ps["obs_state_14"], dtype=np.float32)
            state16 = np.asarray(ps["obs_state_16"], dtype=np.float32)
            if state14.shape[0] != steps or state16.shape[0] != steps:
                raise ValueError(
                    f"load_study: {ps_path} has T={state14.shape[0]}"
                    f"/{state16.shape[0]} (14D/16D) but the Phase-1 npz "
                    f"{p1_path} has T={steps} -- the replay that produced "
                    "this policy-state npz diverged from the recording"
                )

            mass_val = float(np.asarray(p1["mass_kg"]))

            episode_id.extend([f"{condition}/ep_{seed}"] * steps)
            condition_col.extend([condition] * steps)
            seed_chunks.append(np.full(steps, seed, dtype=np.int64))
            step_chunks.append(np.arange(steps, dtype=np.int64))
            mass_kg_chunks.append(np.full(steps, mass_val, dtype=np.float64))
            deficit_chunks.append(deficit_z(p1["commanded_delta"], p1["achieved_eef_delta"]))
            ee_force_chunks.append(np.asarray(p1["ee_force"], dtype=np.float32))
            ee_torque_chunks.append(np.asarray(p1["ee_torque"], dtype=np.float32))
            cfrc_obj_chunks.append(np.asarray(p1["cfrc_obj"], dtype=np.float32))
            commanded_delta_chunks.append(np.asarray(p1["commanded_delta"], dtype=np.float32))
            achieved_eef_delta_chunks.append(np.asarray(p1["achieved_eef_delta"], dtype=np.float32))
            grasped_chunks.append(grasped)
            obj_pos_chunks.append(obj_pos)
            policy_state_14_chunks.append(state14)
            policy_state_16_chunks.append(state16)
            for key in mask_chunks:
                mask_chunks[key].append(masks[key])

    mass_kg_arr = np.concatenate(mass_kg_chunks)

    return {
        "model": model,
        "episode_id": np.array(episode_id),
        "seed": np.concatenate(seed_chunks),
        "condition": np.array(condition_col),
        "step": np.concatenate(step_chunks),
        "mass_kg": mass_kg_arr,
        "mass_log_c": mass_log_c(mass_kg_arr),
        "masks": {key: np.concatenate(chunks) for key, chunks in mask_chunks.items()},
        "deficit_z": np.concatenate(deficit_chunks),
        "ee_force": np.concatenate(ee_force_chunks),
        "ee_torque": np.concatenate(ee_torque_chunks),
        "cfrc_obj": np.concatenate(cfrc_obj_chunks),
        "commanded_delta": np.concatenate(commanded_delta_chunks),
        "achieved_eef_delta": np.concatenate(achieved_eef_delta_chunks),
        "grasped": np.concatenate(grasped_chunks),
        "obj_pos": np.concatenate(obj_pos_chunks),
        "policy_state_14": np.concatenate(policy_state_14_chunks),
        "policy_state_16": np.concatenate(policy_state_16_chunks),
    }

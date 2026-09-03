# Copyright (C) 2026 Xiaomi Corporation.
"""Task 1 (Plan 2, mass-com-xr1): policy-format state replay extraction.

The 210 Phase-1 npz already on disk (``output/mass_variation/phase1/...``)
predate ``recorder.py``'s optional ``policy_state`` channel -- they carry
world-frame ground truth (``eef_pos``, forces, ``obj_pos``, ...) but not
either model's actual POLICY-FACING proprio vector. This script re-derives
BOTH policy formats for every recorded episode of both models' corpora, in
ONE bit-exact replay pass each, by reusing ``render_episode.py``'s replay
scaffold verbatim:

- ``parse_npz_path`` -- (cell, condition, seed) from the npz path.
- ``setup_condition_physics`` -- re-derives the density override from the
  INTENDED condition (``conditions.condition_physics``), NOT the npz's
  measured ``mass_kg`` (that would make the replay circular -- see
  ``render_episode.py``'s module docstring); the measured mass/CoM are
  used only as a post-setup verification assert at ``atol=1e-9``.
- The same success/liftoff_step/grasped/obj_pos bit-exactness gate
  render_episode.py enforces before it will write an mp4: here, before
  this script will write a policy-state npz. A divergence raises
  ``ReplayDivergence`` and writes NOTHING for that episode -- per the
  plan, this is never loosened; the CLI catches it, logs it, records it in
  the manifest's ``divergences``, and moves on to the next episode rather
  than aborting the whole batch.

Per-step state extraction (the part render_episode.py does NOT do):
``entry_mass.run_episode`` computes each model's own state vector
right AFTER ``env.step`` (``policy_state = state_fn(observation)``, fed to
both the model's state queue and the optional recorder channel) -- this
script's replay loop mirrors that exactly, calling BOTH state functions on
every post-step observation regardless of which model actually produced
the episode being replayed (the states are functions of the observation
dict, not of the model that generated the recorded actions -- Task 1's
brief is explicit about this: "extract BOTH state formats in ONE replay
pass per episode, for BOTH models' corpora"):

- ``eval_robocasa365.entry.observation_to_state`` (entry.py:58-71) -- XR1's
  14-D axis-angle state (EEF pos rel base, EEF rot axis-angle, gripper
  qpos, base pos, base rot).
- ``pi05_client.state_from_observation`` -- the openpi-robocasa fork's
  16-D raw-quaternion state (main.py:132-142; see pi05_client.py's own
  module docstring for the exact line citation), REUSED here rather than
  hand-mirrored a second time -- it was already written for the T7 pi0.5
  arm and is exercised by ``test_analysis_dataset.py``'s field-order test.

Rendering / camera resolution
------------------------------
The brief calls for "rendering OFF" to cut extraction cost, since no video
frames are read here -- only ``state.*`` observation keys. Checked against
robocasa's gym wrapper (``robocasa/wrappers/gym_wrapper.py`` ->
``env_utils.create_env``): for this study's env-construction path
(``render_onscreen=False``, matching every other script in this package),
``create_env`` sets ``use_camera_obs=(not render_onscreen)`` == ``True``
UNCONDITIONALLY -- there is no kwarg that reaches ``robosuite.make`` to
disable it (``use_camera_obs`` is already an explicit keyword in
``create_env``'s call to ``robosuite.make``, so passing it again via
``gym.make(..., use_camera_obs=False)`` raises a duplicate-keyword
TypeError), and ``RoboCasaGymEnv``'s own ``enable_render=False`` flag only
zeroes the ALREADY-rendered image array afterward (it does not skip the
underlying MuJoCo camera render call) -- verified by reading
``get_basic_observation``. So cameras are effectively required by the
wrapper, per the brief's own fallback: this script instead shrinks
resolution to ``CAMERA_SIDE_PX`` (128, vs Phase-1's 256) via
``gym.make(..., camera_widths=..., camera_heights=...)``, which DOES
reduce the render cost, since the ``state.*`` keys this script reads are
completely unaffected by camera resolution.

Run under the robocasa venv with EGL, same as ``render_episode.py``:

    MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.extract_policy_state [--limit N] \\
        [--time-budget-s S]

Resumable: an episode whose ``policy_state`` npz already exists, or whose
``(model, condition, seed)`` key is already recorded in the manifest's
``divergences``, is skipped -- so bounded foreground invocations
(``--time-budget-s`` or ``--limit``) can be repeated until the corpus is
complete, mirroring ``run_phase1.py``'s resumability convention.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval_robocasa365.entry import observation_to_state
from eval_robocasa365.mass_variation.conditions import episode_seeds
from eval_robocasa365.mass_variation.entry_mass import PRIMARY_CATEGORY, PRIMARY_ENV
from eval_robocasa365.mass_variation.pi05_client import state_from_observation
from eval_robocasa365.mass_variation.recorder import StepRecorder
from eval_robocasa365.mass_variation.render_episode import (
    COM_OFFSET_ATOL_M,
    DEFAULT_OBJ_NAME,
    MASS_ATOL_KG,
    ReplayDivergence,
    _assert_close,
    parse_npz_path,
    setup_condition_physics,
)

log = logging.getLogger("extract_policy_state")

# See the module docstring's "Rendering / camera resolution" note: cameras
# cannot be disabled for this env-construction path, so this shrinks
# resolution instead (never reading the resulting video.* keys at all).
CAMERA_SIDE_PX = 128
MASS_CONDITIONS = ("MassLight", "MassMedium", "MassHeavy")
MODELS = ("xr1", "pi05_robocasa")


def cell_dir_for(cell: str, model: str) -> str:
    """``run_phase1.cell_dir_for``'s convention, duplicated (not imported):
    ``run_phase1.py`` pulls in server-lifecycle-only stdlib modules
    (subprocess/signal/socket) this offline extraction script has no
    reason to depend on. XR1 keeps the unsuffixed cell dir; every other
    model gets a ``__<model>`` suffix.
    """
    return cell if model == "xr1" else f"{cell}__{model}"


def phase1_npz_path(phase1_root: Path, cell: str, model: str, condition: str, seed: int) -> Path:
    return phase1_root / cell_dir_for(cell, model) / condition / f"ep_{seed}.npz"


def policy_state_npz_path(policy_state_root: Path, model: str, condition: str, seed: int) -> Path:
    return policy_state_root / model / condition / f"ep_{seed}.npz"


def extract_episode_policy_state(
    npz_path: str | Path,
    out_path: str | Path,
    obj_name: str = DEFAULT_OBJ_NAME,
    category: str = PRIMARY_CATEGORY,
    camera_side_px: int = CAMERA_SIDE_PX,
) -> dict[str, Any]:
    """Bit-exact replay ``npz_path`` and write ``out_path`` with
    ``obs_state_14``/``obs_state_16`` -- (T, 14) and (T, 16) arrays, one
    row per recorded step, computed from the post-step observation exactly
    as the original run's ``entry_mass.run_episode`` did (see the module
    docstring). Raises ``ReplayDivergence`` (writes nothing) on any
    success/liftoff_step/grasped/obj_pos mismatch against the recording.
    """
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import convert_action

    npz_path = Path(npz_path)
    out_path = Path(out_path)

    with np.load(npz_path) as data:
        npz = {key: data[key] for key in data.files}

    cell, condition, seed_from_path = parse_npz_path(npz_path)
    seed = int(npz["seed"])
    if seed != seed_from_path:
        raise ValueError(
            f"{npz_path}: seed mismatch -- path implies {seed_from_path}, "
            f"npz 'seed' scalar says {seed}"
        )

    recorded_actions = np.asarray(npz["actions"], dtype=np.float32)
    steps = recorded_actions.shape[0]
    recorded_success = bool(npz["success"])
    recorded_liftoff = int(npz["liftoff_step"])
    recorded_grasped = np.asarray(npz["grasped"], dtype=bool)
    recorded_obj_pos = np.asarray(npz["obj_pos"], dtype=np.float32)
    recorded_mass_kg = float(npz["mass_kg"])
    recorded_com_offset_m = float(npz["com_offset_m"])

    log.info(
        "Extracting %s: cell=%s condition=%s seed=%d steps=%d",
        npz_path, cell, condition, seed, steps,
    )

    env = gym.make(
        f"robocasa/{cell}", split="pretrain", obj_groups=category, seed=seed,
        camera_widths=camera_side_px, camera_heights=camera_side_px,
    )
    try:
        setup = setup_condition_physics(env, condition, seed, obj_name=obj_name)
        _assert_close("mass_kg", setup["mass_kg"], recorded_mass_kg, MASS_ATOL_KG, npz_path)
        _assert_close(
            "com_offset_m", setup["com_offset_m"], recorded_com_offset_m,
            COM_OFFSET_ATOL_M, npz_path,
        )

        recorder = StepRecorder()
        states_14: list[np.ndarray] = []
        states_16: list[np.ndarray] = []
        info: dict[str, Any] = {}
        for t in range(steps):
            observation, _, done, truncated, info = env.step(
                convert_action(recorded_actions[t])
            )
            # Computed ONCE each, from the SAME post-step observation, for
            # BOTH models -- see the module docstring: the states are
            # functions of the obs dict, not of which model this npz's
            # actions came from.
            states_14.append(observation_to_state(observation))
            states_16.append(state_from_observation(observation))
            recorder.record(env, obj_name, recorded_actions[t])

        replayed_success = bool(info.get("success", False))
    finally:
        env.close()

    # Reuse the exact bit-exactness gate render_episode.py enforces: write
    # the recorder's own trace to a throwaway npz (its finalize() derives
    # liftoff_step the same way the ORIGINAL Phase-1 run's recorder did),
    # compare, then discard it -- this script's real output is the state
    # arrays below, written only once the gate passes.
    replay_npz_path = out_path.with_suffix(".replay_check.npz")
    recorder.finalize(
        replay_npz_path,
        mass_kg=setup["mass_kg"], com_offset_m=setup["com_offset_m"],
        com_axis=setup["com_axis"], seed=seed, success=replayed_success,
    )
    try:
        with np.load(replay_npz_path) as replay_data:
            replayed_liftoff = int(replay_data["liftoff_step"])
            replayed_grasped = np.asarray(replay_data["grasped"], dtype=bool)
            replayed_obj_pos = np.asarray(replay_data["obj_pos"], dtype=np.float32)
    finally:
        replay_npz_path.unlink(missing_ok=True)

    divergences = []
    if replayed_success != recorded_success:
        divergences.append(f"success: replayed={replayed_success} recorded={recorded_success}")
    if replayed_liftoff != recorded_liftoff:
        divergences.append(
            f"liftoff_step: replayed={replayed_liftoff} recorded={recorded_liftoff}"
        )
    if replayed_grasped.shape != recorded_grasped.shape or not np.array_equal(
        replayed_grasped, recorded_grasped
    ):
        divergences.append("grasped trace differs from the recording")
    if replayed_obj_pos.shape != recorded_obj_pos.shape or not np.allclose(
        replayed_obj_pos, recorded_obj_pos, atol=1e-4, rtol=0.0
    ):
        max_diff = (
            float(np.max(np.abs(replayed_obj_pos - recorded_obj_pos)))
            if replayed_obj_pos.shape == recorded_obj_pos.shape
            else float("nan")
        )
        divergences.append(f"obj_pos trace differs from the recording (max |diff|={max_diff:.3e} m)")

    if divergences:
        raise ReplayDivergence(f"{npz_path}: replay diverged: " + "; ".join(divergences))

    obs_state_14 = np.stack(states_14).astype(np.float32)
    obs_state_16 = np.stack(states_16).astype(np.float32)
    if obs_state_14.shape != (steps, 14):
        raise ValueError(f"{npz_path}: expected obs_state_14 shape {(steps, 14)}, got {obs_state_14.shape}")
    if obs_state_16.shape != (steps, 16):
        raise ValueError(f"{npz_path}: expected obs_state_16 shape {(steps, 16)}, got {obs_state_16.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        obs_state_14=obs_state_14,
        obs_state_16=obs_state_16,
        seed=np.int64(seed),
        mass_kg=np.float64(setup["mass_kg"]),
        success=np.bool_(replayed_success),
        liftoff_step=np.int64(replayed_liftoff),
    )
    log.info("Wrote %s (%d steps, bit-exact replay verified)", out_path, steps)

    return {
        "npz_path": str(npz_path),
        "out_path": str(out_path),
        "cell": cell,
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "success": replayed_success,
        "liftoff_step": replayed_liftoff,
        "bit_exact": True,
    }


# ---------------------------------------------------------------------------
# CLI: batch over both corpora, resumable, manifest.
# ---------------------------------------------------------------------------


def build_pending(
    phase1_root: Path,
    policy_state_root: Path,
    cell: str,
    models: list[str],
    conditions: list[str],
    seeds: list[int],
    manifest: dict[str, Any],
) -> list[tuple[str, str, int, Path, Path]]:
    """Every ``(model, condition, seed)`` whose Phase-1 npz exists but
    whose policy-state npz does not, EXCLUDING keys already recorded as a
    ``divergences`` entry in ``manifest`` -- a divergence is a property of
    the fixed npz + this script's replay logic, so re-running would just
    reproduce the identical failure; it stays reported, not silently
    retried forever.
    """
    pending = []
    for model in models:
        for condition in conditions:
            for seed in seeds:
                key = f"{model}/{condition}/ep_{seed}"
                if key in manifest.get("divergences", {}):
                    continue
                p1_path = phase1_npz_path(phase1_root, cell, model, condition, seed)
                if not p1_path.exists():
                    continue
                out_path = policy_state_npz_path(policy_state_root, model, condition, seed)
                if out_path.exists():
                    continue
                pending.append((model, condition, seed, p1_path, out_path))
    return pending


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {"episodes": {}, "divergences": {}}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 1 (Plan 2): replay-extract policy-format states for both models."
    )
    parser.add_argument("--phase1-root", default="output/mass_variation/phase1")
    parser.add_argument("--policy-state-root", default="output/mass_variation/policy_state")
    parser.add_argument("--cell", default=PRIMARY_ENV)
    parser.add_argument("--category", default=PRIMARY_CATEGORY)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--conditions", nargs="+", default=list(MASS_CONDITIONS))
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=35)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many pending episodes this invocation (spot checks).",
    )
    parser.add_argument(
        "--time-budget-s", type=float, default=None,
        help="Exit cleanly (resumable) once this much wall time has elapsed.",
    )
    parser.add_argument("--camera-side-px", type=int, default=CAMERA_SIDE_PX)
    parser.add_argument(
        "--manifest", default=None,
        help="Default: <policy-state-root>/policy_state_manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    phase1_root = Path(args.phase1_root)
    policy_state_root = Path(args.policy_state_root)
    manifest_path = (
        Path(args.manifest) if args.manifest else policy_state_root / "policy_state_manifest.json"
    )
    seeds = episode_seeds(args.base_seed, args.cell_index, args.n_seeds)

    manifest = load_manifest(manifest_path)
    pending = build_pending(
        phase1_root, policy_state_root, args.cell, args.models, args.conditions, seeds, manifest,
    )
    log.info("pending=%d (before --limit)", len(pending))
    if args.limit is not None:
        pending = pending[: args.limit]

    t0 = time.monotonic()
    n_ok = 0
    n_div = 0
    for model, condition, seed, p1_path, out_path in pending:
        if args.time_budget_s is not None and (time.monotonic() - t0) >= args.time_budget_s:
            log.info("BUDGET_EXIT: time budget reached; re-invoke to resume.")
            break

        key = f"{model}/{condition}/ep_{seed}"
        ep_t0 = time.monotonic()
        try:
            result = extract_episode_policy_state(
                p1_path, out_path, category=args.category, camera_side_px=args.camera_side_px,
            )
            wall_s = time.monotonic() - ep_t0
            result["model"] = model
            result["wall_s"] = round(wall_s, 1)
            manifest["episodes"][key] = result
            n_ok += 1
            log.info(
                "OK %s: steps=%d success=%s liftoff=%d wall=%.1fs",
                key, result["steps"], result["success"], result["liftoff_step"], wall_s,
            )
        except ReplayDivergence as exc:
            log.error("DIVERGED %s: %s", key, exc)
            manifest["divergences"][key] = {
                "model": model, "condition": condition, "seed": seed,
                "npz_path": str(p1_path), "error": str(exc),
            }
            n_div += 1
        save_manifest(manifest_path, manifest)

    remaining = build_pending(
        phase1_root, policy_state_root, args.cell, args.models, args.conditions, seeds, manifest,
    )
    summary = {
        "status": "complete" if not remaining else "incomplete",
        "extracted_this_run": n_ok,
        "diverged_this_run": n_div,
        "remaining": len(remaining),
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

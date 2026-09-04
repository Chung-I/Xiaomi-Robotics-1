# Copyright (C) 2026 Xiaomi Corporation.
"""Plan-2 Task 2: the five recoverability certificates (the decisive experiment).

Answers, per model, whether hidden mass is recoverable from each observation
channel with this corpus and the study's grouped-CV discipline -- BEFORE any
activation probing is interpreted (a probe null is only meaningful given a
PASSing certificate; the sequential rule lives in the plan's Task 5).

The five certificates (plan Task 2, exact):

1. ``raw_ft``       -- wrist ``ee_force``+``ee_torque`` k=16 trailing windows
                       -> ``mass_log_c``. The physics channel. Both corpora.
                       AMENDED (plan amendment B Section 1): the window input
                       additionally carries physics-derived per-step features
                       |F|, |tau|, world-frame force and base-frame force --
                       see :func:`derived_force_features` for the exact,
                       documented transform. Gate unchanged.
2. ``policy_obs_xr1``  -- ``window_stack(policy_state_14, k=4, stride=2)``
                       flattened (exactly the XR1 policy's 4-frame proprio
                       format, achieved-only) -> ``mass_log_c``. THE headline
                       certificate. XR1 corpus only. Input SACRED (amendment
                       B Section 2: readers changed, inputs not).
3. ``policy_obs_pi05`` -- ``policy_state_16`` single frame -> ``mass_log_c``.
                       The sensing-gap test. pi0.5 corpus only. Input SACRED.
4. ``deficit``      -- ``commanded_delta``+``achieved_eef_delta`` k=8 trailing
                       windows -> ``mass_log_c``. Analysis-side channel
                       content. Both corpora.
5. ``k_eff``        -- AMENDED (amendment B Section 3): two in-carry per-step
                       OLS regressions with documented units: (a) ``|F| ~
                       mass_kg`` (channel validation; slope in N/kg, expect
                       ~g plus load-dynamics inflation) and (b) world-frame
                       ``F_z ~ deficit_z`` (impedance story). UNITS:
                       ``deficit_z = commanded_delta_z - achieved_eef_delta_z``
                       mixes NORMALIZED OSC ACTION UNITS (commanded, std
                       ~0.36-0.51, range +-1) with meters (achieved, std
                       ~0.004-0.005) -- the deficit is therefore dominated by
                       the commanded term and is reported in action units,
                       never silently converted. Not gated.

No-circularity rule (pre-registered, as amended): certificates 2-3 use ONLY
policy-observable state channels (no forces anywhere in their features);
certificate 1 uses force/torque channels plus, per amendment B, the recorded
ORIENTATION chain (``policy_state_14`` dims 3:6 and 11:14) solely to
re-express the same forces -- no position/gripper channels enter it.

Method (ported discipline, via the copied ``probe_core``): ridge with the
probe core's per-fold SVD closed form and alpha grid; CV = GroupKFold(5)
grouped by SEED (35 groups per model -- the matched-pairs design shares one
scene layout across the 3 conditions of a seed, so seed-grouping is what
prevents layout-identity leakage across folds); selectivity = real - mean of
5 group-coherent label shuffles, each with its own alpha search.

Hybrid shuffle groups (documented design choice): the label shuffle is
group-coherent at the EPISODE level (condition/seed), not the seed level.
``mass_log_c`` is constant per episode but VARIES within a seed group (3
conditions per seed), so seed-level shuffling would take ``probe_core``'s
block-swap branch -- and because every seed's rows carry the same
condition-ordered label pattern (light block, medium block, heavy block),
swapping same-offset blocks between seeds is a near no-op that would leave
``shuffled ~= real`` and deflate selectivity to ~0 for genuinely decodable
targets (regression-guarded in ``test_certificates.py``). Episode-level
shuffle groups keep the control an actual permutation of the per-episode
constant labels (probe_core's table branch) while the CV folds stay
seed-grouped.

Gate (pre-registered, before any result was seen): mass R^2 >= 0.3 on the
``carry`` mask per certificate. For the two policy_obs certificates two
neural readers (torch, GPU, seeded) are trained ADDITIONALLY (amendment B
Section 2 -- inputs sacred, readers extended and fairly budgeted):

- a 2x128 ReLU MLP on the certificate's exact (sacred) input format --
  ridge stays reported as the linear floor;
- a 2-layer GRU width 96 over the raw per-step state sequence, a recurrent
  upper bound on the same channel (it tells apart "the channel does not
  carry mass" from "the policy's frame format cannot expose it").

Both are trained MINIBATCHED (the pre-amendment GRU took one full-batch
Adam step per epoch and stopped after 31-88 steps -- reviewer-verified
undertrained), 300-epoch cap, patience 20 on held-out-train R^2. A
certificate PASSes if ANY of its readers clears the gate; every reader's
verdict is reported separately. Neural-reader numbers carry a reliability
note: 29-35 seed groups is a data-starved regime for a trained reader, so
treat them as noisy readers, not tight bounds.

The pre-amendment-B table is preserved verbatim under the output JSON's
``pre_amendment_B`` key (amendment B Section 4).

Run (robocasa venv, GPU free):
    ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.analysis.certificates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from eval_robocasa365.mass_variation.analysis import probe_core
from eval_robocasa365.mass_variation.analysis.dataset import load_study, window_stack

# ----------------------------------------------------------------- constants

SEED = 0
GATE_R2 = 0.3
GATE_MASK = "carry"
CERT_MASKS = ("carry", "all")
K_RAW_FT = 16
K_DEFICIT = 8
K_POLICY = 4
STRIDE_POLICY = 2
DEGENERATE_VAR_TOL = 1e-12

MODELS = ("xr1", "pi05_robocasa")
# certificate -> models it is meaningful for (plan Task 2)
CERT_MODELS = {
    "raw_ft": MODELS,
    "policy_obs_xr1": ("xr1",),
    "policy_obs_pi05": ("pi05_robocasa",),
    "deficit": MODELS,
}
NEURAL_CERTS = ("policy_obs_xr1", "policy_obs_pi05")
NEURAL_KINDS = ("mlp", "gru")  # amendment B Section 2 reader set (ridge = linear floor)
NEURAL_RELIABILITY_NOTE = (
    "trained reader in a data-starved regime (29-35 seed groups, 3 mass "
    "levels): treat as a noisy reader, not a tight bound on channel content"
)

INPUT_CHANNELS = {
    "raw_ft": [
        f"ee_force[3]+ee_torque[3]+|F|+|tau|+F_world[3]+F_base[3] "
        f"(amendment B Section 1 derived features; orientation chain from "
        f"policy_state_14 dims 3:6 & 11:14, rotation use only -- see "
        f"derived_force_features), trailing window k={K_RAW_FT} stride 1"
    ],
    "policy_obs_xr1": [
        f"policy_state_14 (XR1 obs format: EEF pos rel base, EEF rot axis-angle, "
        f"gripper qpos, base pos, base rot), window_stack k={K_POLICY} "
        f"stride {STRIDE_POLICY} (ridge+mlp, sacred format) / raw per-step "
        f"sequence (gru)"
    ],
    "policy_obs_pi05": [
        "policy_state_16 (pi0.5 obs format), single frame (ridge+mlp, sacred "
        "format) / raw per-step sequence (gru)"
    ],
    "deficit": [
        f"commanded_delta[6]+achieved_eef_delta[6], trailing window k={K_DEFICIT} stride 1"
    ],
    "k_eff": [
        "amendment B Section 3: |F| ~ mass_kg and world-frame F_z ~ deficit_z, "
        "per-step OLS on carry; deficit_z in normalized action units (see "
        "K_EFF_UNITS_NOTE)"
    ],
}

K_EFF_UNITS_NOTE = (
    "deficit_z = commanded_delta_z - achieved_eef_delta_z mixes units: "
    "commanded_delta is the raw normalized OSC action (dimensionless, "
    "range ~+-1, carry std ~0.51 xr1 / ~0.36 pi05) while achieved_eef_delta "
    "is metric (carry std ~0.005 m xr1 / ~0.004 m pi05; effective tracking "
    "gain ~0.01 m per action unit), so deficit_z is dominated by the "
    "commanded term and its slopes are reported per NORMALIZED ACTION UNIT, "
    "never silently converted to meters"
)

# Neural-reader hyperparameters (amendment B Section 2 fair budget; the
# worker mirrors these -- it cannot import this sklearn-side module)
GRU_HIDDEN = 96
GRU_LAYERS = 2
MLP_HIDDEN = 128
NEURAL_LR = 1e-3
NEURAL_MAX_EPOCHS = 300
NEURAL_PATIENCE = 20
NEURAL_EP_BATCH = 16   # gru: episodes per gradient step
NEURAL_ROW_BATCH = 256  # mlp: masked rows per gradient step
NEURAL_BUDGET_S = 300.0  # per (certificate, mask, reader), all folds together

DEFAULT_PHASE1_ROOT = "output/mass_variation/phase1"
DEFAULT_POLICY_STATE_ROOT = "output/mass_variation/policy_state"
DEFAULT_OUT = "output/mass_variation/analysis/certificates.json"


# ---------------------------------------------------------- pure (unit-tested)


def axis_angle_to_matrix(aa) -> np.ndarray:
    """Batch Rodrigues: axis-angle vectors ``(..., 3)`` -> rotation matrices
    ``(..., 3, 3)``; the zero vector maps to the identity."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)  # (..., 1)
    small = theta[..., 0] < 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(theta > 1e-12, aa / theta, 0.0)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = theta[..., None]
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    R[small] = np.eye(3)
    return R


def derived_force_features(ee_force, ee_torque, policy_state_14) -> np.ndarray:
    """Amendment B Section 1 physics-derived per-step features: ``(N, 8)`` =
    ``[|F|, |tau|, F_world (3), F_base (3)]``.

    Exact transform (documented per the amendment). The recorder's
    ``ee_force``/``ee_torque`` are the robosuite wrist F/T sensor readings
    in the rotating EE (sensor-site) frame; gravity is world-z, so a linear
    reader on raw components cannot see the load. The recorded orientation
    chain (``policy_state_14``, ``entry.observation_to_state`` field order):

    - ``R_be = axis_angle_to_matrix(policy_state_14[:, 3:6])`` -- EEF
      orientation RELATIVE TO BASE
      (``state.end_effector_rotation_relative``);
    - ``R_wb = axis_angle_to_matrix(policy_state_14[:, 11:14])`` -- base
      orientation in world (``state.base_rotation``; yaw-only in this
      corpus, base upright);
    - ``F_base = R_be @ F_ee`` (base-frame force);
    - ``F_world = R_wb @ F_base`` (world-frame force).

    BOTH rotated variants are included: they differ only by the base yaw --
    constant within an episode but varying ~1.5 rad std ACROSS episodes --
    which is exactly the kind of rotation a single linear reader fit across
    episodes cannot undo. The base frame is the yaw-invariant, gravity-
    aligned choice (base upright => base z == world z); ``F_world`` is the
    amendment's literal item. Corpus diagnostics (``diagnostics.py``) show
    the carried load lands on a FIXED base-frame axis (y), mass-monotone --
    i.e. the sensor site carries a constant offset rotation relative to the
    recorded EEF frame, which a linear reader absorbs. Only orientation
    dims (3:6, 11:14) of ``policy_state_14`` are touched -- no position or
    gripper channels enter the physics certificate (no-circularity as
    amended)."""
    F = np.asarray(ee_force, dtype=np.float64)
    tau = np.asarray(ee_torque, dtype=np.float64)
    ps = np.asarray(policy_state_14, dtype=np.float64)
    if F.shape[1:] != (3,) or tau.shape[1:] != (3,) or ps.shape[1:] != (14,):
        raise ValueError(
            f"derived_force_features: bad shapes {F.shape} {tau.shape} {ps.shape}")
    R_be = axis_angle_to_matrix(ps[:, 3:6])
    R_wb = axis_angle_to_matrix(ps[:, 11:14])
    F_base = (R_be @ F[:, :, None])[..., 0]
    F_world = (R_wb @ F_base[:, :, None])[..., 0]
    return np.concatenate(
        [np.linalg.norm(F, axis=1, keepdims=True),
         np.linalg.norm(tau, axis=1, keepdims=True),
         F_world, F_base], axis=1).astype(np.float32)


def per_episode_windows(x, episode_id, k: int, stride: int) -> np.ndarray:
    """``window_stack`` applied per contiguous episode segment, flattened to
    ``(N, k * C)`` -- windows NEVER cross an episode boundary (each episode's
    early steps left-clamp on its own first frame, exactly like the live
    policy loop restarting its history deque per episode)."""
    x = np.asarray(x)
    episode_id = np.asarray(episode_id)
    if x.shape[0] != episode_id.shape[0]:
        raise ValueError(f"length mismatch: {x.shape[0]} vs {episode_id.shape[0]}")
    # contiguity check: each episode id must form exactly one run
    change = np.flatnonzero(episode_id[1:] != episode_id[:-1])
    n_runs = len(change) + 1
    if n_runs != len(np.unique(episode_id)):
        raise ValueError("per_episode_windows: episode rows are not contiguous")
    starts = np.concatenate([[0], change + 1, [len(episode_id)]])
    blocks = []
    for a, b in zip(starts[:-1], starts[1:]):
        w = window_stack(x[a:b], k=k, stride=stride)
        blocks.append(w.reshape(w.shape[0], -1))
    return np.concatenate(blocks, axis=0)


def rank_acc_levels(y_true, y_pred) -> float:
    """Pairwise ordering accuracy over different-true-level pairs (strict: a
    tied prediction counts as incorrect); chance 0.5; NaN with no valid pairs.

    Same semantics as the origin study's ``rank_accuracy`` with a single
    object (this study's corpus has one object, so same-object == always),
    but O(n log n) per level pair via sorting instead of the O(n^2) pair
    matrix (the ``all`` mask has ~68k rows -> ~2e9 pairs)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    levels = np.unique(y_true)
    correct = total = 0
    preds_by_level = {lv: np.sort(y_pred[y_true == lv]) for lv in levels}
    for i, lo in enumerate(levels):
        for hi in levels[i + 1 :]:
            p_lo, p_hi = preds_by_level[lo], preds_by_level[hi]
            total += len(p_lo) * len(p_hi)
            # correct pair: pred at the higher true level strictly above
            correct += int(np.searchsorted(p_lo, p_hi, side="left").sum())
    return correct / total if total else float("nan")


def degenerate_cell(reason: str) -> dict:
    """The degenerate-guard result (empty mask / constant target): NaN-free
    JSON (None fields) plus a loud flag, per the Global Constraints."""
    return {
        "degenerate": True,
        "degenerate_reason": reason,
        "r2_pooled": None,
        "r2_folds": None,
        "rank_acc": None,
        "shuffled": None,
        "shuffled_std": None,
        "selectivity": None,
        "floor": None,
        "n": 0,
        "n_groups": 0,
    }


def cell_from_factors(factors, splits, y, cv_groups, shuffle_groups,
                      seed: int = SEED) -> dict:
    """THE certificate statistic, given precomputed per-fold ridge factors.

    Split out of :func:`certificate_cell` (behaviour unchanged) so readers
    whose per-fold feature map is not a plain global one -- Plan amendment
    C's vision certificate fits a PCA inside each fold before the ridge --
    can reuse the identical protocol instead of restating it: best-alpha
    pooled held-out R2, per-fold R2s, rank_acc from those same held-out
    predictions, ``probe_core.N_SHUFFLES`` group-coherent label shuffles
    each with its OWN alpha search, and the predict-the-train-mean floor.

    ``factors`` are ``probe_core._fold_factors``-shaped dicts (``tr``,
    ``te``, ``U``, ``s``, ``G``); ``splits`` the matching (train, test)
    index pairs. ``cv_groups`` (seed) drive the folds and the floor;
    ``shuffle_groups`` (episode) drive the shuffle -- see the module
    docstring's "hybrid shuffle groups" note.
    """
    y = np.asarray(y, dtype=np.float64)
    real, pred = probe_core._cv_pooled_best_factored(factors, y, "reg", return_pred=True)
    rng = np.random.default_rng(seed)
    shuf_scores = [
        probe_core._cv_pooled_best_factored(
            factors, probe_core._shuffle_group_coherent(y, shuffle_groups, rng), "reg"
        )
        for _ in range(probe_core.N_SHUFFLES)
    ]
    floor = probe_core._floor(y, np.asarray(cv_groups), "reg", splits=splits)
    return {
        "degenerate": False,
        "r2_pooled": float(real),
        "r2_folds": [float(probe_core._score(y[te], pred[te], "reg")) for _, te in splits],
        "rank_acc": float(rank_acc_levels(y, pred)),
        "shuffled": float(np.mean(shuf_scores)),
        "shuffled_std": float(np.std(shuf_scores)),
        "selectivity": float(real - np.mean(shuf_scores)),
        "floor": float(floor),
        "n": int(len(y)),
        "n_groups": int(len(np.unique(cv_groups))),
    }


def certificate_cell(X, y, cv_groups, shuffle_groups, seed: int = SEED) -> dict:
    """One ridge certificate cell on already-masked rows.

    ``cv_groups`` (seed) drive the GroupKFold(5) folds and the floor;
    ``shuffle_groups`` (episode) drive the group-coherent label shuffle --
    see the module docstring's "hybrid shuffle groups" note. Features are
    globally z-scored (label-free, mixed physical units: Newtons vs radians
    vs meters) before the probe core's centered-SVD ridge path."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return degenerate_cell("empty mask")
    if float(np.var(y)) < DEGENERATE_VAR_TOL:
        return degenerate_cell(f"target variance {float(np.var(y)):.3e} < {DEGENERATE_VAR_TOL}")
    X = np.asarray(X, dtype=np.float32)
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd == 0, 1.0, sd)
    cv_groups = np.asarray(cv_groups)
    shuffle_groups = np.asarray(shuffle_groups)

    splits = probe_core._group_splits(X, cv_groups)
    factors = probe_core._fold_factors(X, cv_groups, splits=splits)
    return cell_from_factors(factors, splits, y, cv_groups, shuffle_groups, seed=seed)


def fit_k_eff(y, x, condition) -> dict:
    """Per-step OLS ``y = slope * x + intercept`` plus per-condition means --
    the certificate-5 regression primitive (physical validation, not gated).
    Generic in (y, x) since amendment B Section 3 reframes k_eff as TWO
    regressions (``|F| ~ mass_kg`` and world-frame ``F_z ~ deficit_z``)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    condition = np.asarray(condition)
    A = np.stack([x, np.ones_like(x)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    per_condition = {}
    for c in np.unique(condition):
        m = condition == c
        sub_a = np.stack([x[m], np.ones(int(m.sum()))], axis=1)
        (c_slope, _), *_ = np.linalg.lstsq(sub_a, y[m], rcond=None)
        per_condition[str(c)] = {
            "y_mean": float(y[m].mean()),
            "x_mean": float(x[m].mean()),
            "slope": float(c_slope),
            "n": int(m.sum()),
        }
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n": int(len(y)),
        "per_condition": per_condition,
    }


# ---------------------------------------------------------- feature builders


def build_features(cert: str, d: dict) -> np.ndarray:
    """The certificate's design matrix over ALL rows (masking happens per
    cell). Enforces the no-circularity rule by construction: each branch
    touches only its pre-registered channels (raw_ft additionally reads
    ``policy_state_14``'s ORIENTATION dims for the amendment-B rotations --
    see ``derived_force_features``; unit-tested no-contamination)."""
    eid = d["episode_id"]
    if cert == "raw_ft":
        ft = np.concatenate(
            [d["ee_force"], d["ee_torque"],  # raw components (pre-amendment set)
             derived_force_features(d["ee_force"], d["ee_torque"],
                                    d["policy_state_14"])],  # amendment B Section 1
            axis=1)
        return per_episode_windows(ft, eid, k=K_RAW_FT, stride=1)
    if cert == "policy_obs_xr1":
        return per_episode_windows(d["policy_state_14"], eid, k=K_POLICY, stride=STRIDE_POLICY)
    if cert == "policy_obs_pi05":
        return np.asarray(d["policy_state_16"])  # single frame, policy states only
    if cert == "deficit":
        delta = np.concatenate([d["commanded_delta"], d["achieved_eef_delta"]], axis=1)
        return per_episode_windows(delta, eid, k=K_DEFICIT, stride=1)
    raise ValueError(f"unknown certificate {cert!r}")


def gru_state_key(cert: str) -> str:
    return {"policy_obs_xr1": "policy_state_14", "policy_obs_pi05": "policy_state_16"}[cert]


# ---------------------------------------------------- neural-reader pipeline


def run_neural_certificate(d: dict, cert: str, mask_name: str, mode: str,
                           gru_python: str, device: str, X=None,
                           max_epochs: int = NEURAL_MAX_EPOCHS,
                           budget_s: float = NEURAL_BUDGET_S,
                           patience: int = NEURAL_PATIENCE) -> dict:
    """One neural-reader cell (amendment B Section 2), ``mode`` in
    {"gru", "mlp"}: the GRU consumes the raw per-step policy-state sequence
    (recurrent upper bound); the MLP consumes ``X``, the certificate's exact
    (sacred) design matrix as built by :func:`build_features`. Loss and
    scoring only on masked rows; same seed-grouped masked-row folds as the
    ridge path; early-stop on one held-out train SEED (its 3 conditions
    give the per-episode-constant target nonzero validation variance);
    minibatched with a 300-epoch cap and patience 20 (fair budget -- the
    pre-amendment GRU was reviewer-verified undertrained).

    Training runs in a SUBPROCESS under ``gru_python``
    (``.venv-mibot/bin/python``: this venv's torch 2.7.1+cu126 has no
    sm_120 kernels for the RTX 5090; mibot's torch 2.8.0+cu128 does), with
    npz interchange -- see ``gru_worker.py``. The fold partition is computed
    HERE (probe_core / sklearn side) and shipped to the worker, so the
    neural readers use byte-identical folds to the ridge path; pooled R2 /
    rank_acc are computed HERE from the worker's returned held-out
    predictions, so the metric implementations stay single-sourced."""
    import subprocess
    import tempfile

    mask_all = d["masks"][mask_name]
    y_all = np.asarray(d["mass_log_c"], dtype=np.float64)
    if mask_all.sum() == 0:
        return degenerate_cell("empty mask")
    if float(np.var(y_all[mask_all])) < DEGENERATE_VAR_TOL:
        return degenerate_cell("constant target under mask")

    eid, seeds = d["episode_id"], d["seed"]

    # identical folds to the ridge path: GroupKFold over MASKED rows by seed
    seeds_masked = seeds[mask_all]
    splits = probe_core._group_splits(np.zeros((int(mask_all.sum()), 1)), seeds_masked)
    folds = [
        (sorted(set(seeds_masked[tr].tolist())), sorted(set(seeds_masked[te].tolist())))
        for tr, te in splits
    ]

    payload = {
        "y": y_all, "mask": mask_all,
        "seed": np.asarray(seeds, dtype=np.int64),
        "folds_json": np.array(json.dumps(folds)),
    }
    if mode == "gru":
        change = np.flatnonzero(eid[1:] != eid[:-1])
        payload["ep_start"] = np.concatenate(
            [[0], change + 1, [len(eid)]]).astype(np.int64)
        payload["states"] = np.asarray(d[gru_state_key(cert)], dtype=np.float32)
    elif mode == "mlp":
        if X is None:
            raise ValueError("run_neural_certificate: mlp mode needs X")
        payload["X"] = np.asarray(X, dtype=np.float32)
    else:
        raise ValueError(f"unknown neural mode {mode!r}")

    with tempfile.TemporaryDirectory(prefix="neural_exchange_") as tmp:
        in_npz = str(Path(tmp) / "in.npz")
        out_npz = str(Path(tmp) / "out.npz")
        np.savez_compressed(in_npz, **payload)
        cmd = [gru_python, "-u", "-m",
               "eval_robocasa365.mass_variation.analysis.gru_worker",
               in_npz, out_npz, "--mode", mode, "--device", device,
               "--max-epochs", str(max_epochs), "--budget-s", str(budget_s),
               "--patience", str(patience),
               "--ep-batch", str(NEURAL_EP_BATCH),
               "--row-batch", str(NEURAL_ROW_BATCH)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parents[3]))
        if proc.returncode != 0:
            raise RuntimeError(
                f"{mode} worker failed for {cert}/{mask_name} (rc={proc.returncode}):\n"
                f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
        sys.stdout.write(proc.stdout)
        with np.load(out_npz) as z:
            res = {k: z[k] for k in z.files}

    yt, yp = res["y_true"], res["y_pred"]
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    r2_pooled = 1.0 - float(((yp - yt) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "degenerate": False,
        "r2_pooled": float(r2_pooled),
        "r2_folds": [float(r) for r in res["r2_folds"]],
        "rank_acc": float(rank_acc_levels(yt, yp)),
        "shuffled": None,  # shuffle control is the ridge path's; neural
        "shuffled_std": None,  # readers are seeded + early-stopped bounds
        "selectivity": None,
        "floor": None,
        "n": int(mask_all.sum()),
        "n_groups": int(len(np.unique(seeds_masked))),
        "epochs_per_fold": [int(e) for e in res["epochs_per_fold"]],
        "budget_hit": bool(res["budget_hit"]),
        "wall_s": float(res["wall_s"]),
        "reliability_note": NEURAL_RELIABILITY_NOTE,
    }


# ----------------------------------------------------------------------- main


def sanitize_json(obj):
    """Non-finite floats -> None, recursively (RFC 8259 has no NaN/Inf)."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def gate_verdict(cells_for_cert: dict) -> dict:
    """PASS/FAIL on the pre-registered gate: mass R^2 >= GATE_R2 on carry.
    ``cells_for_cert`` maps kind ('ridge'/'gru') -> {mask: cell}."""
    entry = {"gate_r2": GATE_R2, "gate_mask": GATE_MASK}
    passes = []
    for kind, by_mask in cells_for_cert.items():
        cell = by_mask.get(GATE_MASK)
        r2 = None if cell is None or cell.get("degenerate") else cell["r2_pooled"]
        ok = bool(r2 is not None and r2 >= GATE_R2)
        entry[f"{kind}_r2"] = r2
        entry[f"{kind}_pass"] = ok
        passes.append(ok)
    entry["pass"] = any(passes)
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1-root", default=DEFAULT_PHASE1_ROOT)
    ap.add_argument("--policy-state-root", default=DEFAULT_POLICY_STATE_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gru-python", default=".venv-mibot/bin/python",
                    help="python with an sm_120-capable torch for the neural "
                         "worker subprocess (see run_neural_certificate)")
    ap.add_argument("--skip-neural", action="store_true",
                    help="skip the MLP and GRU readers (ridge + k_eff only)")
    ap.add_argument("--max-epochs", type=int, default=NEURAL_MAX_EPOCHS)
    ap.add_argument("--budget-s", type=float, default=NEURAL_BUDGET_S)
    ap.add_argument("--patience", type=int, default=NEURAL_PATIENCE)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="plan2-certificates-v2")
    args = ap.parse_args(argv)
    if not Path(args.gru_python).is_absolute():
        # resolve against the repo root (this file's location), not the cwd
        args.gru_python = str(Path(__file__).resolve().parents[3] / args.gru_python)

    cells = []
    k_eff = {}
    gates = {}
    for model in args.models:
        d = load_study(model, args.phase1_root, args.policy_state_root)
        print(f"[load] {model}: N={len(d['step'])} "
              f"masks={{{', '.join(f'{k}:{int(v.sum())}' for k, v in d['masks'].items())}}}",
              flush=True)
        y = d["mass_log_c"]
        for cert, cert_models in CERT_MODELS.items():
            if model not in cert_models:
                continue
            X = build_features(cert, d)
            per_kind = {"ridge": {}}
            for mask_name in CERT_MASKS:
                m = d["masks"][mask_name]
                t0 = time.time()
                cell = certificate_cell(X[m], y[m], d["seed"][m], d["episode_id"][m])
                cell.update(model=model, certificate=cert, kind="ridge", mask=mask_name,
                            input_channels=INPUT_CHANNELS[cert],
                            wall_s=round(time.time() - t0, 1))
                per_kind["ridge"][mask_name] = cell
                cells.append(cell)
                print(f"[ridge] {model}/{cert}/{mask_name}: "
                      f"R2={cell['r2_pooled']} folds={cell['r2_folds']} "
                      f"sel={cell['selectivity']}+-{cell['shuffled_std']} "
                      f"rank_acc={cell['rank_acc']} ({cell['wall_s']}s)", flush=True)
            if cert in NEURAL_CERTS and not args.skip_neural:
                for kind in NEURAL_KINDS:
                    per_kind[kind] = {}
                    for mask_name in CERT_MASKS:
                        cell = run_neural_certificate(
                            d, cert, mask_name, kind, args.gru_python,
                            args.device, X=X if kind == "mlp" else None,
                            max_epochs=args.max_epochs,
                            budget_s=args.budget_s, patience=args.patience)
                        cell.update(model=model, certificate=cert, kind=kind,
                                    mask=mask_name,
                                    input_channels=INPUT_CHANNELS[cert])
                        per_kind[kind][mask_name] = cell
                        cells.append(cell)
                        print(f"[{kind:5s}] {model}/{cert}/{mask_name}: "
                              f"R2={cell['r2_pooled']} folds={cell.get('r2_folds')} "
                              f"rank_acc={cell.get('rank_acc')} "
                              f"epochs={cell.get('epochs_per_fold')} "
                              f"({cell.get('wall_s')}s)", flush=True)
            gates[f"{model}/{cert}"] = gate_verdict(per_kind)

        # certificate 5 (amendment B Section 3): two carry-mask regressions,
        # units documented in K_EFF_UNITS_NOTE; not gated
        carry = d["masks"]["carry"]
        derived = derived_force_features(d["ee_force"], d["ee_torque"],
                                         d["policy_state_14"])
        f_norm = derived[:, 0]      # |F|
        f_world_z = derived[:, 4]   # F_world z-component
        k_eff[model] = {
            "abs_force_vs_mass": fit_k_eff(  # slope in N/kg, x = mass_kg
                f_norm[carry], d["mass_kg"][carry], d["condition"][carry]),
            "world_fz_vs_deficit": fit_k_eff(  # slope in N per action unit
                f_world_z[carry], d["deficit_z"][carry], d["condition"][carry]),
            "units": K_EFF_UNITS_NOTE,
            "deficit_z_carry_std": float(np.std(d["deficit_z"][carry])),
            "commanded_delta_z_carry_std": float(
                np.std(d["commanded_delta"][carry, 2])),
            "achieved_eef_delta_z_carry_std_m": float(
                np.std(d["achieved_eef_delta"][carry, 2])),
        }
        for name, reg in (("abs_force_vs_mass [N/kg]", k_eff[model]["abs_force_vs_mass"]),
                          ("world_fz_vs_deficit [N/action-unit]",
                           k_eff[model]["world_fz_vs_deficit"])):
            print(f"[k_eff] {model} {name}: slope={reg['slope']:.3f} "
                  f"intercept={reg['intercept']:.3f} R2={reg['r2']:.4f} "
                  f"n={reg['n']}", flush=True)
            for c, v in reg["per_condition"].items():
                print(f"        {c}: y_mean={v['y_mean']:.3f} "
                      f"x_mean={v['x_mean']:.4f} slope={v['slope']:.3f} "
                      f"n={v['n']}", flush=True)

    # ------------------------------------------------------------ gate table
    print("\n========= AMENDED (amendment B) PRE-REGISTERED GATE TABLE "
          f"(mass R2 >= {GATE_R2} on '{GATE_MASK}') =========", flush=True)
    for key, g in gates.items():
        parts = [f"{k[:-3]}={g[k]:.4f}" if isinstance(g[k], float) else f"{k[:-3]}=None"
                 for k in g if k.endswith("_r2") and k != "gate_r2"]
        print(f"  {key:38s} {' '.join(parts):40s} "
              f"-> {'PASS' if g['pass'] else 'FAIL'}", flush=True)
    print("=" * 88, flush=True)

    config = {
        "seed": SEED, "gate_r2": GATE_R2, "gate_mask": GATE_MASK,
        "cert_masks": list(CERT_MASKS),
        "k_raw_ft": K_RAW_FT, "k_deficit": K_DEFICIT,
        "k_policy": K_POLICY, "stride_policy": STRIDE_POLICY,
        "alphas": probe_core.ALPHAS.tolist(), "n_splits": probe_core.N_SPLITS,
        "n_shuffles": probe_core.N_SHUFFLES,
        "cv_groups": "seed (35 matched-pair groups)",
        "shuffle_groups": "episode (condition/seed) -- see module docstring",
        "degenerate_var_tol": DEGENERATE_VAR_TOL,
        "amendment": ("B (2026-09-03): raw_ft derived features |F|/|tau|/"
                      "F_world/F_base; policy_obs readers += seeded 2x128 MLP "
                      "(inputs sacred); GRU/MLP fairly budgeted (minibatched, "
                      "300-epoch cap, patience 20); k_eff reframed as "
                      "|F|~mass and world F_z~deficit_z with documented "
                      "units; pre-amendment table under pre_amendment_B"),
        "neural": {"gru_hidden": GRU_HIDDEN, "gru_layers": GRU_LAYERS,
                   "mlp_hidden": MLP_HIDDEN, "lr": NEURAL_LR,
                   "max_epochs": args.max_epochs, "patience": args.patience,
                   "ep_batch": NEURAL_EP_BATCH, "row_batch": NEURAL_ROW_BATCH,
                   "budget_s": args.budget_s, "certs": list(NEURAL_CERTS),
                   "kinds": list(NEURAL_KINDS),
                   "skipped": bool(args.skip_neural),
                   "reliability_note": NEURAL_RELIABILITY_NOTE,
                   "runner": f"gru_worker.py subprocess via {args.gru_python}"},
        "k_eff_units": K_EFF_UNITS_NOTE,
        "no_circularity_rule": ("certs 2-3 use ONLY policy-state channels; "
                                "cert 1 uses ee_force/ee_torque plus the "
                                "policy_state_14 ORIENTATION dims (3:6, "
                                "11:14) for the amendment-B rotations only"),
        "models": list(args.models),
        "phase1_root": args.phase1_root,
        "policy_state_root": args.policy_state_root,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    try:
        import sklearn
        config["versions"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import torch
        config["versions"]["torch"] = torch.__version__
    except ImportError:
        pass
    if not args.skip_neural:
        import subprocess
        proc = subprocess.run(
            [args.gru_python, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True)
        if proc.returncode == 0:
            config["versions"]["neural_torch"] = proc.stdout.strip()

    # amendment B Section 4: preserve the pre-amendment table for the record.
    # If the existing output file is itself pre-amendment (no marker), stash
    # its {config, gates, k_eff, cells}; if it already carries the stash
    # (an amended re-run), carry that stash through unchanged.
    pre_amendment = None
    out_path = Path(args.out)
    if out_path.exists():
        try:
            old = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            old = None
        if old is not None:
            if "pre_amendment_B" in old:
                pre_amendment = old["pre_amendment_B"]
            elif "amendment" not in old.get("config", {}):
                pre_amendment = {k: old.get(k)
                                 for k in ("config", "gates", "k_eff", "cells")}

    out = sanitize_json({"config": config, "gates": gates, "k_eff": k_eff,
                         "cells": cells, "pre_amendment_B": pre_amendment})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_path} "
          f"(pre_amendment_B {'preserved' if pre_amendment else 'ABSENT'})",
          flush=True)

    if not args.no_wandb:
        import pandas as pd
        import wandb
        run = wandb.init(project="mass-com-xr1", job_type="analysis",
                         name=args.wandb_name, config=config)
        cell_rows = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                      for k, v in c.items()} for c in cells]
        run.log({"certificates": wandb.Table(dataframe=pd.DataFrame(cell_rows))})
        run.summary.update({f"gate/{key}/{k}": v for key, g in gates.items()
                            for k, v in g.items() if v is not None})
        run.summary.update({
            f"k_eff/{m}/{reg_name}/{k}": v
            for m, ke in k_eff.items()
            for reg_name in ("abs_force_vs_mass", "world_fz_vs_deficit")
            for k, v in ke[reg_name].items() if not isinstance(v, dict)})
        print("wandb url:", run.url, flush=True)
        run.finish()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

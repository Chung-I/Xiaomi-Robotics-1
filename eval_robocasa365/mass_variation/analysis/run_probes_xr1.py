# Copyright (C) 2026 Xiaomi Corporation.
"""Plan-2 Task 4: probe XR1's captured activations (the study's final experiment).

Joins the Task-3 activation corpus (``output/mass_variation/activations/xr1``,
per-episode npz at REPLAN steps -- the capture contract in that root's
``meta.json``) with the Task-1 labels/masks at exactly those steps, and runs
the ported probe discipline over the full grid:

- targets: ``mass_log_c`` (primary, reg), ``wrench_norm`` (|F| of
  ``ee_force``, reg), ``deficit_z`` (reg), ``grasped`` (clf),
  ``step_clock`` (= step / T per episode, reg -- the decodability-ceiling
  control). The plan's object-mesh identity control is SKIPPED AND NOTED:
  the Phase-1 npz record no mesh/instance id (``ep_meta`` was not captured),
  and the corpus is a single object category by design (one cell), so the
  control would be constant anyway.
- feature blocks: VLM layers {0, 7, 14, 21, 28, 35} (plan amendment A;
  capture holds all 36) x 4 prefix positions [last_prefix_token + 3
  per-camera image-token means]; DiT taps at the same 6 layer indices x
  {flow 0, flow 4} x {state-block mean, action-block mean}; plus the
  ``state_embed`` block (state_projector output, flattened 4x1024 -- the
  contract's 'state_token' position, since this checkpoint's VLM prefix
  carries no state token).
- masks: {precontact, carry, all} evaluated at the captured steps.

Method (same discipline as ``certificates.py``, via the copied
``probe_core``): ridge/logistic on per-fold centered-SVD factors, CV =
GroupKFold(5) grouped by SEED (35 matched-pair groups); label shuffles are
group-coherent at the EPISODE level (the hybrid documented in
``certificates.py`` -- seed-level shuffling of a mass label that varies
within a seed group would take probe_core's block-swap branch and deflate
selectivity to ~0 for genuinely decodable targets); selectivity = real -
mean of 5 per-draw-alpha-searched shuffles; rank_acc (reg only) from the
real fit's pooled held-out predictions. Features are NOT z-scored
(homogeneous activation units; matches the origin study's ``sweep``), and
degenerate cells (empty mask, constant target, single-class target, < 5
seed groups, a class missing from some training fold) are flagged, never
silently scored.

Sanity gates (pre-registered, abort loudly):

1. ceiling analog -- ``step_clock`` real R^2 > 0.9 at SOME (layer,
   position) on mask ``all``;
2. leakage guard -- ``mass_log_c`` selectivity < 0.1 at EVERY probed cell
   on ``precontact`` (before first contact NOTHING observable correlates
   with the hidden mass; a violation means leakage and the run reports
   BLOCKED -- per the plan the guard is never relaxed).

Outputs: ``<out-dir>/<out-name>`` parquet (one row per grid cell),
``<out-dir>/summary.json`` (gate verdicts + headline), figures under
``<out-dir>/figures/``, wandb run ``plan2-probes-xr1`` (project
``mass-com-xr1``).

Run (robocasa venv; pure analysis, no GPU):

    ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.analysis.run_probes_xr1

Random-init bound (plan Task 4 Step 2 + amendment A key layers): after
capturing ``activations/xr1_random`` with ``replay_capture --random-init``
over the seeds 7-11 x 3 conditions subset, probe the same cells with::

    ... run_probes_xr1 --acts-root output/mass_variation/activations/xr1_random \\
        --vlm-layers 0,14,28,35 --dit-layers 0,14,28,35 --n-seeds 5 \\
        --out-name results_random.parquet --tag random --skip-gates

and build the trained-vs-random table with ``--compare``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval_robocasa365.mass_variation.analysis import probe_core
from eval_robocasa365.mass_variation.analysis.certificates import rank_acc_levels
from eval_robocasa365.mass_variation.analysis.dataset import load_study

REPO_ROOT = Path(__file__).resolve().parents[3]

# Plan amendment A: probe 6 of the 36 captured VLM layers, evenly by depth;
# DiT taps at the same indices; random-init bound at the 4 key layers.
PROBE_VLM_LAYERS = (0, 7, 14, 21, 28, 35)
PROBE_DIT_LAYERS = (0, 7, 14, 21, 28, 35)
RANDOM_KEY_LAYERS = (0, 14, 28, 35)

# Capture-contract facts (asserted against each npz at load).
CAPTURED_VLM_LAYERS = 36
CAPTURED_DIT_LAYERS = 36
VLM_POSITION_NAMES = (
    "last_prefix_token",
    "image_tokens_mean:video.robot0_agentview_left",
    "image_tokens_mean:video.robot0_agentview_right",
    "image_tokens_mean:video.robot0_eye_in_hand",
)
DIT_FLOW_STEPS = (0, 4)
DIT_POSITION_NAMES = (  # flow-step x block, in (flow, block) axis order
    "flow0:state_tokens_mean",
    "flow0:action_tokens_mean",
    "flow4:state_tokens_mean",
    "flow4:action_tokens_mean",
)

MASK_NAMES = ("precontact", "carry", "all")
TARGET_TASKS = {  # 5 targets; mesh-identity control skipped (module docstring)
    "mass_log_c": "reg",
    "wrench_norm": "reg",
    "deficit_z": "reg",
    "grasped": "clf",
    "step_clock": "reg",
}
MESH_CONTROL_NOTE = (
    "object-mesh identity control SKIPPED: Phase-1 npz record no mesh/ep_meta "
    "id and the corpus is one object category by design (constant control)"
)

SEED = 0
GATE_STEP_CLOCK_R2 = 0.9
GATE_LEAKAGE_SELECTIVITY = 0.1
DEGENERATE_VAR_TOL = 1e-12
MIN_GROUPS = probe_core.N_SPLITS

MASS_CONDITIONS = ("MassLight", "MassMedium", "MassHeavy")

DEFAULT_ACTS_ROOT = "output/mass_variation/activations/xr1"
DEFAULT_PHASE1_ROOT = "output/mass_variation/phase1"
DEFAULT_POLICY_STATE_ROOT = "output/mass_variation/policy_state"
DEFAULT_OUT_DIR = "output/mass_variation/analysis/probes_xr1"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in test_run_probes_xr1.py)
# ---------------------------------------------------------------------------


def join_index(ds_episode_id, ds_step, acts_episode_id, acts_step) -> np.ndarray:
    """Dataset-row index for each acts row, keyed on ``(episode_id, step)``.

    ``ds_episode_id``/``ds_step`` are ``load_study``'s per-step columns
    (steps 0..T-1 per episode); ``acts_episode_id``/``acts_step`` name the
    captured replan steps. Raises ``ValueError`` on duplicate dataset keys
    and ``KeyError`` on a capture step with no dataset row (an off-by-one
    or corpus mismatch must be loud, never a silently misaligned label)."""
    ds_episode_id = np.asarray(ds_episode_id)
    ds_step = np.asarray(ds_step)
    acts_episode_id = np.asarray(acts_episode_id)
    acts_step = np.asarray(acts_step)
    table: dict[tuple[str, int], int] = {}
    for row, (eid, step) in enumerate(zip(ds_episode_id, ds_step)):
        key = (str(eid), int(step))
        if key in table:
            raise ValueError(f"join_index: duplicate dataset row for {key}")
        table[key] = row
    idx = np.empty(len(acts_episode_id), dtype=np.int64)
    for i, (eid, step) in enumerate(zip(acts_episode_id, acts_step)):
        key = (str(eid), int(step))
        if key not in table:
            raise KeyError(
                f"join_index: no dataset row for episode {key[0]!r} step {key[1]}"
            )
        idx[i] = table[key]
    return idx


def step_clock(episode_id, step, total_steps: dict[str, int] | None = None) -> np.ndarray:
    """The control target ``step / T`` per episode, in ``[0, 1)``.

    ``T`` is the episode's TOTAL recorded step count: inferred as
    ``max(step) + 1`` per episode when called on full per-step rows (the
    ``load_study`` layout), or passed explicitly via ``total_steps`` when
    called on an already-joined subset (replan steps only -- inferring T
    from a subset's max would inflate early clocks)."""
    episode_id = np.asarray(episode_id)
    step = np.asarray(step, dtype=np.float64)
    if total_steps is None:
        total_steps = {}
        for eid in np.unique(episode_id):
            total_steps[str(eid)] = int(step[episode_id == eid].max()) + 1
    T = np.array([float(total_steps[str(eid)]) for eid in episode_id])
    return step / T


def wrench_norm_of(ee_force) -> np.ndarray:
    """|F| = the Euclidean norm of the (N, 3) wrist ``ee_force`` channel."""
    F = np.asarray(ee_force, dtype=np.float64)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"wrench_norm_of: expected (N, 3) ee_force, got {F.shape}")
    return np.linalg.norm(F, axis=1)


def iter_feature_blocks(acts: dict):
    """Yield ``(block, layer_id, position_name, X (M, D) float32)`` over the
    capture contract's arrays: ``vlm (M, L_v, 4, 2560)`` + ``vlm_layer_ids``,
    ``dit (M, L_d, 2, 2, 1024)`` (flow axis = captured {0, 4}) +
    ``dit_layer_ids``, ``state_embed (M, 4, 1024)`` (one block, flattened,
    ``layer_id`` -1). ``layer_id`` is the REAL checkpoint layer index (the
    array axis holds only the probed subset)."""
    vlm = acts["vlm"]
    for li, layer_id in enumerate(acts["vlm_layer_ids"]):
        for pi, pos in enumerate(VLM_POSITION_NAMES):
            yield "vlm", int(layer_id), pos, vlm[:, li, pi, :].astype(np.float32)
    dit = acts["dit"]
    for li, layer_id in enumerate(acts["dit_layer_ids"]):
        for fi, _flow in enumerate(DIT_FLOW_STEPS):
            for bi in range(2):
                pos = DIT_POSITION_NAMES[fi * 2 + bi]
                yield "dit", int(layer_id), pos, dit[:, li, fi, bi, :].astype(np.float32)
    se = acts["state_embed"]
    yield "state_embed", -1, "state_tokens_flat", se.reshape(se.shape[0], -1).astype(np.float32)


def _degenerate_cell(reason: str, n: int = 0, n_groups: int = 0) -> dict:
    return {
        "degenerate": True,
        "degenerate_reason": reason,
        "real": None,
        "shuffled": None,
        "shuffled_std": None,
        "floor": None,
        "selectivity": None,
        "rank_acc": None,
        "n": int(n),
        "n_groups": int(n_groups),
    }


def _degeneracy_reason(y, cv_groups, task: str, splits=None) -> str | None:
    """The pre-flight degenerate guard (Global Constraints + thin cells)."""
    y = np.asarray(y)
    if len(y) == 0:
        return "empty mask"
    n_groups = len(np.unique(cv_groups))
    if n_groups < MIN_GROUPS:
        return f"only {n_groups} groups < {MIN_GROUPS} (GroupKFold)"
    if task == "reg":
        var = float(np.var(np.asarray(y, dtype=np.float64)))
        if var < DEGENERATE_VAR_TOL:
            return f"target variance {var:.3e} < {DEGENERATE_VAR_TOL}"
    else:
        if len(np.unique(y)) < 2:
            return "single class under mask"
        if splits is not None:
            for k, (tr, _te) in enumerate(splits):
                if len(np.unique(y[tr])) < 2:
                    return f"class missing from fold {k} training rows"
    return None


def _cell_from_factors(
    factors, y, cv_groups, shuffle_groups, task: str, seed: int = SEED
) -> dict:
    """One probe cell on precomputed fold factors, with the study's HYBRID
    shuffle (CV folds/floor grouped by ``cv_groups`` = seed; label shuffle
    group-coherent over ``shuffle_groups`` = episode -- see the module
    docstring). Mirrors ``probe_core._cell_from_factors``'s statistics; adds
    ``rank_acc`` (reg) from the real fit's pooled held-out predictions."""
    y = np.asarray(y)
    splits = [(f["tr"], f["te"]) for f in factors]
    reason = _degeneracy_reason(y, cv_groups, task, splits=splits)
    if reason is not None:
        return _degenerate_cell(reason, n=len(y), n_groups=len(np.unique(cv_groups)))
    rng = np.random.default_rng(seed)
    real, pred = probe_core._cv_pooled_best_factored(factors, y, task, return_pred=True)
    shuf_scores = [
        probe_core._cv_pooled_best_factored(
            factors, probe_core._shuffle_group_coherent(y, np.asarray(shuffle_groups), rng), task
        )
        for _ in range(probe_core.N_SHUFFLES)
    ]
    floor = probe_core._floor(y, np.asarray(cv_groups), task, splits=splits)
    return {
        "degenerate": False,
        "degenerate_reason": None,
        "real": float(real),
        "shuffled": float(np.mean(shuf_scores)),
        "shuffled_std": float(np.std(shuf_scores)),
        "floor": float(floor),
        "selectivity": float(real - np.mean(shuf_scores)),
        "rank_acc": float(rank_acc_levels(y, pred)) if task == "reg" else None,
        "n": int(len(y)),
        "n_groups": int(len(np.unique(cv_groups))),
    }


def probe_target_cell(X, y, cv_groups, shuffle_groups, task: str = "reg", seed: int = SEED) -> dict:
    """Single-cell entry point (tests + spot checks): degenerate guards,
    then factors, then :func:`_cell_from_factors`."""
    y = np.asarray(y)
    cv_groups = np.asarray(cv_groups)
    reason = _degeneracy_reason(y, cv_groups, task)
    if reason is not None:
        return _degenerate_cell(reason, n=len(y), n_groups=len(np.unique(cv_groups)))
    X = np.asarray(X, dtype=np.float32)
    splits = probe_core._group_splits(X, cv_groups)
    factors = probe_core._fold_factors(X, cv_groups, splits=splits)
    return _cell_from_factors(factors, y, cv_groups, shuffle_groups, task, seed)


# ---------------------------------------------------------------------------
# Acts + label loading
# ---------------------------------------------------------------------------


def load_acts(
    acts_root: Path,
    conditions=MASS_CONDITIONS,
    seeds=None,
    vlm_layer_ids=PROBE_VLM_LAYERS,
    dit_layer_ids=PROBE_DIT_LAYERS,
) -> dict:
    """Concatenate the per-episode acts npz (probed-layer slices only) in
    ``(condition, seed)`` order, with per-row ``episode_id``/``step``.
    Aborts loudly on a missing episode or non-finite activations (a
    random-init capture could overflow f16 -- that must be visible, not
    silently probed)."""
    acts_root = Path(acts_root)
    if seeds is None:
        seeds = range(7, 42)
    vlm_chunks, dit_chunks, se_chunks = [], [], []
    eid_col, step_col = [], []
    n_nonfinite = 0
    for condition in conditions:
        for seed in seeds:
            path = acts_root / condition / f"ep_{seed}.npz"
            if not path.exists():
                raise FileNotFoundError(f"load_acts: missing {path}")
            with np.load(path) as data:
                steps = np.asarray(data["steps"], dtype=np.int64)
                vlm = data["vlm"]
                dit = data["dit"]
                se = data["state_embed"]
            if vlm.shape[1] != CAPTURED_VLM_LAYERS or dit.shape[1] != CAPTURED_DIT_LAYERS:
                raise ValueError(f"{path}: unexpected layer axes {vlm.shape} {dit.shape}")
            vlm = vlm[:, list(vlm_layer_ids)]
            dit = dit[:, list(dit_layer_ids)]
            for arr in (vlm, dit, se):
                n_nonfinite += int((~np.isfinite(arr.astype(np.float32))).sum())
            vlm_chunks.append(vlm)
            dit_chunks.append(dit)
            se_chunks.append(se)
            eid = f"{condition}/ep_{seed}"
            eid_col.extend([eid] * len(steps))
            step_col.append(steps)
    if n_nonfinite:
        raise ValueError(
            f"load_acts: {n_nonfinite} non-finite activation values under "
            f"{acts_root} -- refusing to probe (f16 overflow? inspect the capture)"
        )
    return {
        "vlm": np.concatenate(vlm_chunks),
        "vlm_layer_ids": tuple(vlm_layer_ids),
        "dit": np.concatenate(dit_chunks),
        "dit_layer_ids": tuple(dit_layer_ids),
        "state_embed": np.concatenate(se_chunks),
        "episode_id": np.array(eid_col),
        "step": np.concatenate(step_col),
    }


def build_labels(ds: dict, idx: np.ndarray) -> dict:
    """Targets + masks + groups at the joined (replan-step) rows."""
    return {
        "targets": {
            "mass_log_c": np.asarray(ds["mass_log_c"])[idx],
            "wrench_norm": wrench_norm_of(ds["ee_force"])[idx],
            "deficit_z": np.asarray(ds["deficit_z"], dtype=np.float64)[idx],
            "grasped": np.asarray(ds["grasped"], dtype=bool)[idx],
            # computed on FULL per-step rows first, then subset (per-episode
            # T must come from the full episode, not the replan subset)
            "step_clock": step_clock(ds["episode_id"], ds["step"])[idx],
        },
        "masks": {m: np.asarray(ds["masks"][m], dtype=bool)[idx] for m in MASK_NAMES},
        "cv_groups": np.asarray(ds["seed"])[idx],
        "shuffle_groups": np.asarray(ds["episode_id"])[idx],
    }


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------


def _block_cells(
    block: str, layer: int, position: str, X: np.ndarray,
    labels: dict, splits_by_mask: dict, seed: int,
) -> list[dict]:
    """All (target, mask) cells for one feature block. SVD factors are
    computed once per (block, mask) and shared across targets and shuffle
    draws -- ``probe_core.sweep``'s structure with the hybrid shuffle
    groups. Pure function of its arguments (each cell seeds its own rng),
    so blocks can run in parallel workers without changing any statistic."""
    masks = labels["masks"]
    cv_groups = labels["cv_groups"]
    shuffle_groups = labels["shuffle_groups"]
    rows = []
    for mname, mask in masks.items():
        factors = None
        for tname, task in TARGET_TASKS.items():
            y = labels["targets"][tname]
            if splits_by_mask[mname] is None:
                cell = _degenerate_cell(
                    "empty mask or too few groups",
                    n=int(mask.sum()),
                    n_groups=len(np.unique(cv_groups[mask])),
                )
            else:
                reason = _degeneracy_reason(y[mask], cv_groups[mask], task)
                if reason is not None:
                    cell = _degenerate_cell(
                        reason, n=int(mask.sum()),
                        n_groups=len(np.unique(cv_groups[mask])),
                    )
                else:
                    if factors is None:
                        factors = probe_core._fold_factors(
                            X[mask], cv_groups[mask], splits=splits_by_mask[mname]
                        )
                    cell = _cell_from_factors(
                        factors, y[mask], cv_groups[mask],
                        shuffle_groups[mask], task, seed,
                    )
            rows.append({
                "target": tname, "task": task, "block": block,
                "layer": layer, "position": position, "mask": mname, **cell,
            })
    return rows


class GridBudgetExit(RuntimeError):
    """Raised when the time budget expires with blocks still pending (the
    per-block cache holds everything finished; re-invoke to resume)."""


def _block_key(block: str, layer: int, position: str) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in position)
    return f"{block}_L{layer}_{safe}"


def run_grid(
    acts: dict,
    labels: dict,
    seed: int = SEED,
    verbose: bool = True,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    time_budget_s: float | None = None,
) -> pd.DataFrame:
    """The full (target x block x layer x position x mask) grid; blocks run
    in ``n_jobs``-sized joblib batches (independent, deterministic -- row
    order and every statistic identical to the serial run). With
    ``cache_dir`` each finished block's rows are parqueted immediately and
    skipped on re-invocation, so a ``time_budget_s``-bounded foreground run
    is resumable (raises :class:`GridBudgetExit` when the budget expires
    with blocks pending)."""
    masks = labels["masks"]
    cv_groups = labels["cv_groups"]
    splits_by_mask = {}
    for mname, mask in masks.items():
        n = int(mask.sum())
        if n and len(np.unique(cv_groups[mask])) >= MIN_GROUPS:
            splits_by_mask[mname] = probe_core._group_splits(
                np.zeros((n, 1)), cv_groups[mask]
            )
        else:
            splits_by_mask[mname] = None

    blocks = list(iter_feature_blocks(acts))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, pd.DataFrame] = {}
    pending = []
    for block, layer, position, X in blocks:
        key = _block_key(block, layer, position)
        cache_file = None if cache_dir is None else cache_dir / f"{key}.parquet"
        if cache_file is not None and cache_file.exists():
            done[key] = pd.read_parquet(cache_file)
        else:
            pending.append((key, cache_file, block, layer, position, X))
    if verbose and done:
        print(f"[grid] resumed {len(done)}/{len(blocks)} blocks from {cache_dir}",
              flush=True)

    t0 = time.time()
    from joblib import Parallel, delayed

    batch_size = max(1, n_jobs)
    for start in range(0, len(pending), batch_size):
        if time_budget_s is not None and (time.time() - t0) >= time_budget_s:
            raise GridBudgetExit(
                f"time budget {time_budget_s:.0f}s reached with "
                f"{len(pending) - start} blocks pending -- re-invoke to resume"
            )
        batch = pending[start : start + batch_size]
        if n_jobs == 1:
            batch_rows = [
                _block_cells(b, l, p, X, labels, splits_by_mask, seed)
                for _key, _cf, b, l, p, X in batch
            ]
        else:
            # Backend choice (debugged 2026-09-03/04, both failure modes
            # reproduced): (a) loky with its default 2-BLAS-threads-per-
            # worker OpenBLAS threw "SVD did not converge" on blocks that
            # factor cleanly in-process (zero failures serially over all
            # 147 (block, mask) factorizations); (b) prefer="threads"
            # spin-thrashed OpenBLAS's global lock -- a 19s subset grid ran
            # 25 min without finishing one block. So: loky with workers
            # pinned to SINGLE-threaded BLAS (the well-tested LAPACK path),
            # and any block that still raises LinAlgError is recomputed
            # serially in the parent (full BLAS, the known-good path).
            # Statistics are unaffected -- only where/how the same linear
            # algebra runs.
            from joblib import parallel_config

            def _safe(b, l, p, X):
                try:
                    return _block_cells(b, l, p, X, labels, splits_by_mask, seed)
                except np.linalg.LinAlgError as exc:
                    return {"__linalg_error__": str(exc)}

            with parallel_config(backend="loky", inner_max_num_threads=1):
                batch_rows = Parallel(n_jobs=n_jobs)(
                    delayed(_safe)(b, l, p, X)
                    for _key, _cf, b, l, p, X in batch
                )
            for i, res in enumerate(batch_rows):
                if isinstance(res, dict) and "__linalg_error__" in res:
                    _key, _cf, b, l, p, X = batch[i]
                    if verbose:
                        print(f"[grid] {b} L{l} {p}: worker LinAlgError "
                              f"({res['__linalg_error__']}); recomputing "
                              f"in-parent", flush=True)
                    batch_rows[i] = _block_cells(b, l, p, X, labels,
                                                 splits_by_mask, seed)
        for (key, cache_file, *_), rows in zip(batch, batch_rows):
            frame = pd.DataFrame(rows)
            if cache_file is not None:
                frame.to_parquet(cache_file, index=False)
            done[key] = frame
        if verbose:
            print(f"[grid] {len(done)}/{len(blocks)} blocks "
                  f"({time.time() - t0:6.1f}s elapsed, n_jobs={n_jobs})", flush=True)

    # canonical order = iter_feature_blocks order
    ordered = [done[_block_key(b, l, p)] for b, l, p, _ in blocks]
    return pd.concat(ordered, ignore_index=True)


# ---------------------------------------------------------------------------
# Sanity gates (pre-registered; abort loudly)
# ---------------------------------------------------------------------------


def evaluate_gates(df: pd.DataFrame) -> dict:
    clock = df[(df["target"] == "step_clock") & (df["mask"] == "all") & ~df["degenerate"]]
    best = clock.loc[clock["real"].idxmax()] if len(clock) else None
    gate1 = {
        "name": "step_clock decodes (ceiling analog)",
        "rule": f"real R^2 > {GATE_STEP_CLOCK_R2} at SOME (layer, position) on mask 'all'",
        "best_r2": None if best is None else float(best["real"]),
        "best_cell": None if best is None else
            f"{best['block']}/L{best['layer']}/{best['position']}",
        "pass": bool(best is not None and best["real"] > GATE_STEP_CLOCK_R2),
    }
    leak = df[
        (df["target"] == "mass_log_c") & (df["mask"] == "precontact") & ~df["degenerate"]
    ]
    worst = leak.loc[leak["selectivity"].idxmax()] if len(leak) else None
    gate2 = {
        "name": "mass precontact leakage guard",
        "rule": f"mass_log_c selectivity < {GATE_LEAKAGE_SELECTIVITY} at EVERY probed cell on 'precontact'",
        "max_selectivity": None if worst is None else float(worst["selectivity"]),
        "worst_cell": None if worst is None else
            f"{worst['block']}/L{worst['layer']}/{worst['position']}",
        "n_cells": int(len(leak)),
        "n_violations": int((leak["selectivity"] >= GATE_LEAKAGE_SELECTIVITY).sum()),
        "pass": bool(len(leak) > 0 and
                     (leak["selectivity"] < GATE_LEAKAGE_SELECTIVITY).all()),
    }
    return {"step_clock_ceiling": gate1, "precontact_leakage": gate2}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def write_figures(df: pd.DataFrame, fig_dir: Path, tag: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # CVD-validated categorical palette (dataviz six-checks, light surface);
    # per-series markers double as the secondary encoding.
    palette = {"precontact": "#8055aa", "carry": "#c14f2e", "all": "#3a6ea5"}
    markers = {"precontact": "^", "carry": "o", "all": "s"}

    def one_chart(sub: pd.DataFrame, block: str, target: str, fname: str, title: str):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        for mname in MASK_NAMES:
            cells = sub[(sub["mask"] == mname) & ~sub["degenerate"]]
            if not len(cells):
                continue
            # per layer: best position (the standard "is it decodable at
            # this depth" reading); positions are reported in the parquet
            by_layer = cells.groupby("layer")["real"].max()
            ax.plot(by_layer.index, by_layer.values, marker=markers[mname],
                    ms=6, lw=2.0, color=palette[mname], label=mname)
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel(f"{block.upper()} layer")
        ax.set_ylabel("held-out R² (best position)")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(sorted(sub["layer"].unique()))
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / fname)
        plt.close(fig)
        written.append(str(fig_dir / fname))

    for target in ("mass_log_c", "wrench_norm"):
        for block in ("vlm", "dit"):
            sub = df[(df["target"] == target) & (df["block"] == block)]
            if not len(sub):
                continue
            one_chart(
                sub, block, target, f"r2_vs_layer_{block}_{target}_{tag}.png",
                f"{target} — {block.upper()} probes ({tag})",
            )

    # headline detail: mass_log_c on carry, per VLM position
    sub = df[(df["target"] == "mass_log_c") & (df["block"] == "vlm")
             & (df["mask"] == "carry") & ~df["degenerate"]]
    if len(sub):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        colors = ["#3a6ea5", "#c14f2e", "#2f9e63", "#8055aa"]
        pos_markers = ["s", "o", "D", "^"]
        for pos, color, mk in zip(VLM_POSITION_NAMES, colors, pos_markers):
            cells = sub[sub["position"] == pos].sort_values("layer")
            label = pos.replace("image_tokens_mean:video.robot0_", "img:")
            ax.plot(cells["layer"], cells["real"], marker=mk, ms=5.5, lw=2.0,
                    color=color, label=label)
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("VLM layer")
        ax.set_ylabel("held-out R²")
        ax.set_title(f"mass_log_c on carry, per VLM position ({tag})", fontsize=11)
        ax.set_xticks(sorted(sub["layer"].unique()))
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fname = fig_dir / f"mass_log_c_carry_vlm_positions_{tag}.png"
        fig.savefig(fname)
        plt.close(fig)
        written.append(str(fname))
    return written


# ---------------------------------------------------------------------------
# Trained-vs-random comparison (plan Task 4 Step 2)
# ---------------------------------------------------------------------------


def compare_trained_random(trained_path: Path, random_path: Path) -> pd.DataFrame:
    """Per-cell trained-vs-random table on the cells BOTH parquets probed
    (the amendment-A key layers over the matched episode subset)."""
    keys = ["target", "block", "layer", "position", "mask"]
    t = pd.read_parquet(trained_path)
    r = pd.read_parquet(random_path)
    merged = t.merge(r, on=keys, suffixes=("_trained", "_random"))
    cols = keys + [
        "real_trained", "real_random", "selectivity_trained", "selectivity_random",
        "rank_acc_trained", "rank_acc_random",
        "degenerate_trained", "degenerate_random", "n_trained", "n_random",
    ]
    merged = merged[cols].copy()
    merged["delta_real"] = merged["real_trained"] - merged["real_random"]
    return merged


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def headline_table(df: pd.DataFrame) -> list[dict]:
    """mass_log_c on carry, per layer (best position per block): R^2,
    selectivity +- std, rank_acc -- the study's answer row."""
    sub = df[(df["target"] == "mass_log_c") & (df["mask"] == "carry") & ~df["degenerate"]]
    out = []
    for (block, layer), cells in sub.groupby(["block", "layer"]):
        best = cells.loc[cells["real"].idxmax()]
        out.append({
            "block": block, "layer": int(layer), "position": best["position"],
            "r2": float(best["real"]),
            "selectivity": float(best["selectivity"]),
            "shuffled_std": float(best["shuffled_std"]),
            "rank_acc": float(best["rank_acc"]),
            "n": int(best["n"]),
        })
    return sorted(out, key=lambda r: (r["block"], r["layer"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plan-2 Task 4: XR1 activation probes.")
    ap.add_argument("--acts-root", default=DEFAULT_ACTS_ROOT)
    ap.add_argument("--phase1-root", default=DEFAULT_PHASE1_ROOT)
    ap.add_argument("--policy-state-root", default=DEFAULT_POLICY_STATE_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-name", default="results.parquet")
    ap.add_argument("--vlm-layers", default=",".join(map(str, PROBE_VLM_LAYERS)))
    ap.add_argument("--dit-layers", default=",".join(map(str, PROBE_DIT_LAYERS)))
    ap.add_argument("--base-seed", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=35)
    ap.add_argument("--tag", default="trained",
                    help="row tag + figure suffix (trained / trained_subset / random)")
    ap.add_argument("--skip-gates", action="store_true",
                    help="report gate values without aborting (random-init runs; "
                         "the pre-registered gates bind the trained 6-layer run)")
    ap.add_argument("--time-budget-s", type=float, default=None,
                    help="stop starting new block batches after this many "
                         "seconds (exit 3, per-block cache resumes the run)")
    ap.add_argument("--jobs", type=int, default=10,
                    help="joblib workers over feature blocks (statistics "
                         "identical to serial; 20-core box, BLAS shares the rest)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="plan2-probes-xr1")
    ap.add_argument("--compare", nargs=2, metavar=("TRAINED_PARQUET", "RANDOM_PARQUET"),
                    default=None,
                    help="skip probing; write the trained-vs-random table for two "
                         "existing results parquets (matched cells only)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        merged = compare_trained_random(Path(args.compare[0]), Path(args.compare[1]))
        cmp_path = out_dir / "trained_vs_random.parquet"
        merged.to_parquet(cmp_path, index=False)
        core = merged[(merged["target"] == "mass_log_c") & (merged["mask"] == "carry")]
        print(f"wrote {cmp_path} ({len(merged)} matched cells)")
        with pd.option_context("display.width", 200):
            print(core.to_string(index=False))
        return 0

    vlm_layers = tuple(int(x) for x in args.vlm_layers.split(","))
    dit_layers = tuple(int(x) for x in args.dit_layers.split(","))
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    t0 = time.time()
    print(f"[load] acts from {args.acts_root} (vlm layers {vlm_layers}, "
          f"dit layers {dit_layers}, seeds {seeds[0]}..{seeds[-1]})", flush=True)
    acts = load_acts(Path(args.acts_root), seeds=seeds,
                     vlm_layer_ids=vlm_layers, dit_layer_ids=dit_layers)
    print(f"[load] acts rows M={len(acts['step'])} "
          f"vlm={acts['vlm'].shape} dit={acts['dit'].shape} "
          f"state_embed={acts['state_embed'].shape} ({time.time()-t0:.1f}s)", flush=True)

    ds = load_study("xr1", args.phase1_root, args.policy_state_root, seeds=seeds)
    idx = join_index(ds["episode_id"], ds["step"], acts["episode_id"], acts["step"])
    labels = build_labels(ds, idx)
    mask_counts = {m: int(v.sum()) for m, v in labels["masks"].items()}
    print(f"[join] {len(idx)} rows matched; masks at replan steps: {mask_counts}; "
          f"{MESH_CONTROL_NOTE}", flush=True)

    cache_dir = out_dir / f"block_cache_{args.tag}"
    try:
        df = run_grid(acts, labels, seed=SEED, n_jobs=args.jobs,
                      cache_dir=cache_dir, time_budget_s=args.time_budget_s)
    except GridBudgetExit as exc:
        print(f"BUDGET_EXIT: {exc}", flush=True)
        return 3
    df.insert(0, "tag", args.tag)
    df["acts_root"] = args.acts_root

    out_path = out_dir / args.out_name
    df.to_parquet(out_path, index=False)
    print(f"[out] wrote {out_path} ({len(df)} cells)", flush=True)

    gates = evaluate_gates(df)
    print("\n========= SANITY GATES =========", flush=True)
    for key, g in gates.items():
        print(f"  {key}: {'PASS' if g['pass'] else 'FAIL'} -- "
              + json.dumps(sanitize_json({k: v for k, v in g.items() if k != 'name'})),
              flush=True)
    print("=" * 32, flush=True)

    headline = headline_table(df)
    print("\nHEADLINE (mass_log_c on carry, best position per block/layer):", flush=True)
    for row in headline:
        print(f"  {row['block']:>11s} L{row['layer']:>2} {row['position']:<50s} "
              f"R2={row['r2']:.4f} sel={row['selectivity']:.4f}"
              f"+-{row['shuffled_std']:.4f} rank_acc={row['rank_acc']:.4f} n={row['n']}",
              flush=True)

    figures = []
    if not args.no_figures:
        figures = write_figures(df, out_dir / "figures", args.tag)
        print(f"[fig] wrote {len(figures)} figures under {out_dir / 'figures'}", flush=True)

    config = {
        "plan": "2026-09-03-mass-com-xr1-plan-2-probing task 4",
        "tag": args.tag,
        "acts_root": args.acts_root,
        "vlm_layers": list(vlm_layers), "dit_layers": list(dit_layers),
        "dit_flow_steps": list(DIT_FLOW_STEPS),
        "vlm_positions": list(VLM_POSITION_NAMES),
        "dit_positions": list(DIT_POSITION_NAMES),
        "targets": {k: v for k, v in TARGET_TASKS.items()},
        "mesh_control": MESH_CONTROL_NOTE,
        "masks": list(MASK_NAMES),
        "seeds": [seeds[0], seeds[-1]],
        "rows_probed": int(len(idx)),
        "mask_counts": mask_counts,
        "cv_groups": "seed", "shuffle_groups": "episode (hybrid, certificates.py note)",
        "alphas": probe_core.ALPHAS.tolist(),
        "n_splits": probe_core.N_SPLITS, "n_shuffles": probe_core.N_SHUFFLES,
        "z_scored": False,
        "seed": SEED,
        "n_jobs": args.jobs,
        "gates": {"step_clock_r2": GATE_STEP_CLOCK_R2,
                  "leakage_selectivity": GATE_LEAKAGE_SELECTIVITY,
                  "enforced": not args.skip_gates},
        "git_sha": _git_sha(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    try:
        import sklearn
        config["versions"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass

    summary_path = out_dir / f"summary_{args.tag}.json"
    summary = sanitize_json({
        "config": config,
        "gates": gates,
        "headline_mass_log_c_carry": headline,
        "results_parquet": str(out_path),
        "figures": figures,
        "wall_s": round(time.time() - t0, 1),
    })
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"[out] wrote {summary_path}", flush=True)

    if not args.no_wandb:
        import wandb
        run = wandb.init(project="mass-com-xr1", job_type="analysis",
                         name=f"{args.wandb_name}-{args.tag}"
                              if args.tag != "trained" else args.wandb_name,
                         config=config)
        table_df = df.copy()
        for col in table_df.columns:
            if table_df[col].dtype == object:
                table_df[col] = table_df[col].astype(str)
        run.log({"probe_grid": wandb.Table(dataframe=table_df)})
        run.summary.update({f"gate/{k}/pass": g["pass"] for k, g in gates.items()})
        run.summary.update({
            "gate/step_clock_ceiling/best_r2": gates["step_clock_ceiling"]["best_r2"],
            "gate/precontact_leakage/max_selectivity":
                gates["precontact_leakage"]["max_selectivity"],
        })
        for row in headline:
            key = f"headline/{row['block']}/L{row['layer']}"
            run.summary[f"{key}/r2"] = row["r2"]
            run.summary[f"{key}/selectivity"] = row["selectivity"]
            run.summary[f"{key}/rank_acc"] = row["rank_acc"]
        print("wandb url:", run.url, flush=True)
        run.finish()

    if not args.skip_gates:
        failed = [k for k, g in gates.items() if not g["pass"]]
        if failed:
            print(f"\nBLOCKED: sanity gate(s) {failed} FAILED -- see the gate "
                  f"values above; per the plan the guard is never relaxed.",
                  flush=True)
            return 1
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

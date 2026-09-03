# Copyright (C) 2026 Xiaomi Corporation.
"""Plan-2 Task 2: the five recoverability certificates (the decisive experiment).

Answers, per model, whether hidden mass is recoverable from each observation
channel with this corpus and the study's grouped-CV discipline -- BEFORE any
activation probing is interpreted (a probe null is only meaningful given a
PASSing certificate; the sequential rule lives in the plan's Task 5).

The five certificates (plan Task 2, exact):

1. ``raw_ft``       -- wrist ``ee_force``+``ee_torque`` k=16 trailing windows
                       -> ``mass_log_c``. The physics channel. Both corpora.
2. ``policy_obs_xr1``  -- ``window_stack(policy_state_14, k=4, stride=2)``
                       flattened (exactly the XR1 policy's 4-frame proprio
                       format, achieved-only) -> ``mass_log_c``. THE headline
                       certificate. XR1 corpus only.
3. ``policy_obs_pi05`` -- ``policy_state_16`` single frame -> ``mass_log_c``.
                       The sensing-gap test. pi0.5 corpus only.
4. ``deficit``      -- ``commanded_delta``+``achieved_eef_delta`` k=8 trailing
                       windows -> ``mass_log_c``. Analysis-side channel
                       content. Both corpora.
5. ``k_eff``        -- in-carry per-step OLS ``F_z ~ deficit_z`` per model:
                       slope (K_eff), R^2, per-condition means. Physical
                       validation, NOT gated. Also the empirical arbiter of
                       the T1-review frame question (recorder ``eef_pos``
                       base- vs world-frame): a strongly linear positive-R^2
                       fit supports same-frame deficit pairing.

No-circularity rule (pre-registered): certificates 2-3 use ONLY
policy-observable state channels (no forces anywhere in their features);
certificate 1 uses ONLY force/torque (no policy states).

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
``carry`` mask per certificate. For the two policy_obs certificates a small
GRU (torch, GPU, seeded) is trained ADDITIONALLY: the ridge consumes the
policy's exact observation format; the GRU consumes the raw per-step state
sequence as a recurrent upper bound on the same channel (it tells apart "the
channel does not carry mass" from "the policy's frame format cannot expose
it"). A certificate PASSes if ridge or (where run) GRU clears the gate; both
verdicts are reported separately.

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
GRU_CERTS = ("policy_obs_xr1", "policy_obs_pi05")

INPUT_CHANNELS = {
    "raw_ft": [f"ee_force[3]+ee_torque[3], trailing window k={K_RAW_FT} stride 1"],
    "policy_obs_xr1": [
        f"policy_state_14 (XR1 obs format: EEF pos rel base, EEF rot axis-angle, "
        f"gripper qpos, base pos, base rot), window_stack k={K_POLICY} "
        f"stride {STRIDE_POLICY} (ridge) / raw per-step sequence (gru)"
    ],
    "policy_obs_pi05": [
        "policy_state_16 (pi0.5 obs format), single frame (ridge) / "
        "raw per-step sequence (gru)"
    ],
    "deficit": [
        f"commanded_delta[6]+achieved_eef_delta[6], trailing window k={K_DEFICIT} stride 1"
    ],
    "k_eff": ["ee_force[z] ~ deficit_z, per-step OLS on carry"],
}

# GRU hyperparameters (frozen before results were seen; origin-study discipline)
GRU_HIDDEN = 96
GRU_LAYERS = 2
GRU_LR = 1e-3
GRU_MAX_EPOCHS = 300
GRU_PATIENCE = 30
GRU_BUDGET_S = 300.0  # per (certificate, mask), all folds together

DEFAULT_PHASE1_ROOT = "output/mass_variation/phase1"
DEFAULT_POLICY_STATE_ROOT = "output/mass_variation/policy_state"
DEFAULT_OUT = "output/mass_variation/analysis/certificates.json"


# ---------------------------------------------------------- pure (unit-tested)


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
    real, pred = probe_core._cv_pooled_best_factored(factors, y, "reg", return_pred=True)
    rng = np.random.default_rng(seed)
    shuf_scores = [
        probe_core._cv_pooled_best_factored(
            factors, probe_core._shuffle_group_coherent(y, shuffle_groups, rng), "reg"
        )
        for _ in range(probe_core.N_SHUFFLES)
    ]
    floor = probe_core._floor(y, cv_groups, "reg", splits=splits)
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


def fit_k_eff(fz, deficit, condition) -> dict:
    """Per-step OLS ``F_z = slope * deficit_z + intercept`` plus per-condition
    means -- certificate 5 (physical validation, not gated)."""
    fz = np.asarray(fz, dtype=np.float64)
    deficit = np.asarray(deficit, dtype=np.float64)
    condition = np.asarray(condition)
    A = np.stack([deficit, np.ones_like(deficit)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(A, fz, rcond=None)
    pred = slope * deficit + intercept
    ss_res = float(((fz - pred) ** 2).sum())
    ss_tot = float(((fz - fz.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    per_condition = {}
    for c in np.unique(condition):
        m = condition == c
        sub_a = np.stack([deficit[m], np.ones(int(m.sum()))], axis=1)
        (c_slope, _), *_ = np.linalg.lstsq(sub_a, fz[m], rcond=None)
        per_condition[str(c)] = {
            "fz_mean": float(fz[m].mean()),
            "deficit_z_mean": float(deficit[m].mean()),
            "slope": float(c_slope),
            "n": int(m.sum()),
        }
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n": int(len(fz)),
        "per_condition": per_condition,
    }


# ---------------------------------------------------------- feature builders


def build_features(cert: str, d: dict) -> np.ndarray:
    """The certificate's design matrix over ALL rows (masking happens per
    cell). Enforces the no-circularity rule by construction: each branch
    touches only its pre-registered channels."""
    eid = d["episode_id"]
    if cert == "raw_ft":
        ft = np.concatenate([d["ee_force"], d["ee_torque"]], axis=1)  # forces only
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


# -------------------------------------------------------------- GRU pipeline


def run_gru_certificate(d: dict, cert: str, mask_name: str, gru_python: str,
                        device: str, max_epochs: int = GRU_MAX_EPOCHS,
                        budget_s: float = GRU_BUDGET_S) -> dict:
    """Small seeded GRU on the raw per-step policy-state sequence; loss and
    scoring only on masked steps; same seed-grouped masked-row folds as the
    ridge path; early-stop on one held-out train SEED's episodes (its 3
    conditions give the per-episode-constant target nonzero validation
    variance).

    Training runs in a SUBPROCESS under ``gru_python``
    (``.venv-mibot/bin/python``: this venv's torch 2.7.1+cu126 has no
    sm_120 kernels for the RTX 5090; mibot's torch 2.8.0+cu128 does), with
    npz interchange -- see ``gru_worker.py``. The fold partition is computed
    HERE (probe_core / sklearn side) and shipped to the worker, so the GRU
    uses byte-identical folds to the ridge path; pooled R2 / rank_acc are
    computed HERE from the worker's returned held-out predictions, so the
    metric implementations stay single-sourced."""
    import subprocess
    import tempfile

    mask_all = d["masks"][mask_name]
    y_all = np.asarray(d["mass_log_c"], dtype=np.float64)
    if mask_all.sum() == 0:
        return degenerate_cell("empty mask")
    if float(np.var(y_all[mask_all])) < DEGENERATE_VAR_TOL:
        return degenerate_cell("constant target under mask")

    states = np.asarray(d[gru_state_key(cert)], dtype=np.float32)
    eid, seeds = d["episode_id"], d["seed"]

    # contiguous episode boundaries (study row order)
    change = np.flatnonzero(eid[1:] != eid[:-1])
    ep_start = np.concatenate([[0], change + 1, [len(eid)]]).astype(np.int64)

    # identical folds to the ridge path: GroupKFold over MASKED rows by seed
    seeds_masked = seeds[mask_all]
    splits = probe_core._group_splits(np.zeros((int(mask_all.sum()), 1)), seeds_masked)
    folds = [
        (sorted(set(seeds_masked[tr].tolist())), sorted(set(seeds_masked[te].tolist())))
        for tr, te in splits
    ]

    with tempfile.TemporaryDirectory(prefix="gru_exchange_") as tmp:
        in_npz = str(Path(tmp) / "in.npz")
        out_npz = str(Path(tmp) / "out.npz")
        np.savez_compressed(
            in_npz, states=states, y=y_all, mask=mask_all,
            seed=np.asarray(seeds, dtype=np.int64), ep_start=ep_start,
            folds_json=np.array(json.dumps(folds)),
        )
        cmd = [gru_python, "-u", "-m",
               "eval_robocasa365.mass_variation.analysis.gru_worker",
               in_npz, out_npz, "--device", device,
               "--max-epochs", str(max_epochs), "--budget-s", str(budget_s)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parents[3]))
        if proc.returncode != 0:
            raise RuntimeError(
                f"gru_worker failed for {cert}/{mask_name} (rc={proc.returncode}):\n"
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
        "shuffled": None,  # shuffle control is the ridge path's; GRU is the
        "shuffled_std": None,  # recurrent upper bound, seeded + early-stopped
        "selectivity": None,
        "floor": None,
        "n": int(mask_all.sum()),
        "n_groups": int(len(np.unique(seeds_masked))),
        "epochs_per_fold": [int(e) for e in res["epochs_per_fold"]],
        "budget_hit": bool(res["budget_hit"]),
        "wall_s": float(res["wall_s"]),
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
                    help="python with an sm_120-capable torch for the GRU "
                         "worker subprocess (see run_gru_certificate)")
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=GRU_MAX_EPOCHS)
    ap.add_argument("--budget-s", type=float, default=GRU_BUDGET_S)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="plan2-certificates")
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
            if cert in GRU_CERTS and not args.skip_gru:
                per_kind["gru"] = {}
                for mask_name in CERT_MASKS:
                    cell = run_gru_certificate(d, cert, mask_name,
                                               args.gru_python, args.device,
                                               max_epochs=args.max_epochs,
                                               budget_s=args.budget_s)
                    cell.update(model=model, certificate=cert, kind="gru",
                                mask=mask_name, input_channels=INPUT_CHANNELS[cert])
                    per_kind["gru"][mask_name] = cell
                    cells.append(cell)
                    print(f"[gru]   {model}/{cert}/{mask_name}: "
                          f"R2={cell['r2_pooled']} folds={cell.get('r2_folds')} "
                          f"rank_acc={cell.get('rank_acc')} "
                          f"epochs={cell.get('epochs_per_fold')} "
                          f"({cell.get('wall_s')}s)", flush=True)
            gates[f"{model}/{cert}"] = gate_verdict(per_kind)

        # certificate 5: K_eff (carry-only, not gated)
        carry = d["masks"]["carry"]
        k_eff[model] = fit_k_eff(d["ee_force"][carry, 2], d["deficit_z"][carry],
                                 d["condition"][carry])
        ke = k_eff[model]
        print(f"[k_eff] {model}: slope={ke['slope']:.3f} intercept={ke['intercept']:.3f} "
              f"R2={ke['r2']:.4f} n={ke['n']}", flush=True)
        for c, v in ke["per_condition"].items():
            print(f"        {c}: Fz_mean={v['fz_mean']:.3f} "
                  f"deficit_z_mean={v['deficit_z_mean']:.4f} slope={v['slope']:.3f} "
                  f"n={v['n']}", flush=True)

    # ------------------------------------------------------------ gate table
    print("\n=============== PRE-REGISTERED GATE TABLE "
          f"(mass R2 >= {GATE_R2} on '{GATE_MASK}') ===============", flush=True)
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
        "gru": {"hidden": GRU_HIDDEN, "layers": GRU_LAYERS, "lr": GRU_LR,
                "max_epochs": args.max_epochs, "patience": GRU_PATIENCE,
                "budget_s": args.budget_s, "certs": list(GRU_CERTS),
                "skipped": bool(args.skip_gru),
                "runner": f"gru_worker.py subprocess via {args.gru_python}"},
        "no_circularity_rule": ("certs 2-3 use ONLY policy-state channels; "
                                "cert 1 uses only ee_force/ee_torque"),
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
    if not args.skip_gru:
        import subprocess
        proc = subprocess.run(
            [args.gru_python, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True)
        if proc.returncode == 0:
            config["versions"]["gru_torch"] = proc.stdout.strip()

    out = sanitize_json({"config": config, "gates": gates, "k_eff": k_eff,
                         "cells": cells})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_path}", flush=True)

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
        run.summary.update({f"k_eff/{m}/{k}": v for m, ke in k_eff.items()
                            for k, v in ke.items() if not isinstance(v, dict)})
        print("wandb url:", run.url, flush=True)
        run.finish()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

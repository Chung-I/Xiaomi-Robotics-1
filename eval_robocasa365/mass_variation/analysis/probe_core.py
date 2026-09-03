# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ORIGIN NOTE (XR1 mass/CoM Plan-2, Task 2): this file is COPIED VERBATIM
# (per the plan's "copy, do not reimplement, do not import across repos"
# rule) from the sibling pi0.5/RoboLab probing study:
#   ~/Codes/RoboLab/.claude/worktrees/mass-com-vla-probing/analysis/mass_com/probe_core.py
# (our own code; Apache-2.0, original SPDX header retained above). Only this
# header note was added -- the module has no intra-package imports, so no
# import adaptation was needed. Its behavioral semantics (grouped-CV ridge/
# logistic, group-coherent shuffles with per-draw alpha search, selectivity,
# floor) are pinned by the 4 ported tests in
# ``eval_robocasa365/mass_variation/test_probe_core_port.py``.
"""Pure-numpy/sklearn probe machinery (Plan-3 Task 1).

``run_probe_cell`` is the one linear-probe primitive everything else calls:
grouped-CV ridge/logistic (manual, since sklearn's ``RidgeCV`` has no group
awareness) against three controls per Global Constraints [adj 4] — a
group-coherent label shuffle (Hewitt & Liang's control probe, adapted so
per-episode-constant targets stay per-episode-constant under the shuffle), a
predict-the-mean/majority floor, and the resulting ``selectivity = real -
shuffled``.

Shuffle rule: if the target is constant within every group (the common case —
mass, CoM are per-episode), permute the group->label table so each group gets
another group's constant label. If the target varies within a group (e.g. a
per-step phase clock), whole-group *blocks* of labels are swapped between
groups instead of a per-sample shuffle, which would destroy the group-coherent
temporal structure the control is meant to preserve; group lengths are padded
(by repeating the last value) or trimmed to fit, since groups need not be
equal length in general even though the fixtures here are. This block swap
assumes each group's rows are already in time order (contiguous, step-
ascending) in ``X``/``y``/``groups`` — true of every caller in this study
(the probe dataset and the synthetic fixtures are built that way) — since it
copies same-offset runs of values between groups without re-sorting by step.

``shuffled`` re-searches ALPHAS independently for each of N_SHUFFLES
permutation draws (never reuses ``real``'s alpha): fixing the null to the
signal's alpha is anti-conservative, since a permuted label has no real
structure for a low (weakly regularized) alpha to overfit *to* on the test
fold, other than fold-partition noise — exactly the kind of noise a
low-alpha fit is best at chasing. Letting each draw pick its own best alpha
gives the null its fair (and typically higher, more regularized) optimum,
so ``shuffled`` isn't deflated and ``selectivity`` isn't inflated.

Performance (Task-3 authorized optimizations; no statistic changed):
the ridge path is solved in closed form from one economy SVD of each
training fold — ``X_c = U S V^T`` gives ``w(alpha) = V diag(s/(s^2+alpha))
U^T y_c`` — so all ALPHAS, all N_SHUFFLES draws, and (via ``sweep``) all
targets sharing an (X, mask) reuse the same five per-fold factorizations.
GroupKFold split indices are computed once per (groups, mask). The logistic
path fits sklearn's ``LogisticRegression`` unchanged, but on the rotated
features ``Z = X_c V`` (an orthonormal change of basis, under which the L2
penalty and hence the optimization problem are mathematically identical,
while the dimension drops from D to <= n_train). ``sweep``'s ``task`` may be
a single string (as before) or a dict ``{target_name: task}``.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold

ALPHAS = 10.0 ** np.arange(-2, 5)  # 1e-2 .. 1e4
N_SPLITS = 5
N_SHUFFLES = 5  # permutation draws averaged into the `shuffled` control


def _is_group_constant(y, groups):
    return all(np.allclose(y[groups == g], y[groups == g][0]) for g in np.unique(groups))


def _shuffle_group_coherent(y, groups, rng):
    """Group-coherent label permutation; see module docstring."""
    uniq = np.unique(groups)
    perm = dict(zip(uniq, rng.permutation(uniq)))
    y_shuf = np.empty_like(y)
    if _is_group_constant(y, groups):
        table = {g: y[groups == g][0] for g in uniq}
        for g in uniq:
            y_shuf[groups == g] = table[perm[g]]
        return y_shuf
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    for g in uniq:
        dst, src = idx_by_group[g], idx_by_group[perm[g]]
        n = min(len(dst), len(src))
        y_shuf[dst[:n]] = y[src[:n]]
        if len(dst) > n:  # dst longer than src: pad by repeating src's last value
            y_shuf[dst[n:]] = y[src[n - 1]]
    return y_shuf


def _score(y_true, y_pred, task):
    return r2_score(y_true, y_pred) if task == "reg" else balanced_accuracy_score(y_true, y_pred)


def _group_splits(X, groups):
    return list(GroupKFold(n_splits=N_SPLITS).split(X, np.zeros(len(groups)), groups))


def _fold_factors(X, groups, splits=None):
    """Per-fold economy SVD of the centered training block, plus the test
    block projected into the right-singular basis.

    Each entry holds ``tr, te`` (row indices), ``mu`` (train column means),
    ``U (n_tr, k), s (k,)`` from ``X_tr - mu = U S V^T``, and
    ``G = (X_te - mu) V`` — everything ridge/logistic need, with V itself
    (k x D) discarded. Ridge closed form: pooled test predictions at alpha
    are ``G @ (s/(s^2+alpha) * (U^T y_c)) + y_mean`` — identical to
    ``sklearn.Ridge(alpha, fit_intercept=True)``. Logistic: fit on
    ``Z_tr = U*s = X_c V`` and predict on ``G`` — an orthonormal rotation of
    the feature space, under which sklearn's L2-penalized objective is
    unchanged.
    """
    if splits is None:
        splits = _group_splits(X, groups)
    factors = []
    for tr, te in splits:
        mu = X[tr].mean(axis=0)
        U, s, Vt = np.linalg.svd((X[tr] - mu).astype(np.float64), full_matrices=False)
        G = (X[te] - mu).astype(np.float64) @ Vt.T
        factors.append({"tr": tr, "te": te, "U": U, "s": s, "G": G})
    return factors


def _cv_pooled_best_factored(factors, y, task, return_pred=False):
    """Best pooled held-out score over ALPHAS, from precomputed factors.
    With ``return_pred`` also returns the best alpha's pooled predictions."""
    if task == "reg":
        y = np.asarray(y, dtype=np.float64)
        preds = np.empty((len(ALPHAS), len(y)), dtype=np.float64)
        for f in factors:
            y_tr = y[f["tr"]]
            y_mean = y_tr.mean()
            c = f["U"].T @ (y_tr - y_mean)
            for i, alpha in enumerate(ALPHAS):
                shrink = f["s"] / (f["s"] ** 2 + alpha)
                preds[i, f["te"]] = f["G"] @ (shrink * c) + y_mean
        scores = [_score(y, preds[i], task) for i in range(len(ALPHAS))]
        best_i = int(np.argmax(scores))
        return (scores[best_i], preds[best_i]) if return_pred else scores[best_i]
    best, best_pred = -np.inf, None
    for alpha in ALPHAS:
        pred = np.empty(len(y), dtype=np.asarray(y).dtype)
        for f in factors:
            if "Ztr" not in f:
                f["Ztr"] = f["U"] * f["s"]
            model = LogisticRegression(C=1.0 / alpha, max_iter=1000, class_weight="balanced")
            model.fit(f["Ztr"], y[f["tr"]])
            pred[f["te"]] = model.predict(f["G"])
        score = _score(y, pred, task)
        if score > best:
            best, best_pred = score, pred
    return (best, best_pred) if return_pred else best


def _floor(y, groups, task, splits=None):
    """Predict-the-training-fold-mean (reg) / -majority-class (clf) score."""
    if splits is None:
        splits = _group_splits(np.zeros((len(y), 1)), groups)
    pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in splits:
        if task == "reg":
            pred[te] = y[tr].mean()
        else:
            vals, counts = np.unique(y[tr], return_counts=True)
            pred[te] = vals[np.argmax(counts)]
    return _score(y, pred, task)


def _cell_from_factors(factors, y, groups, task, seed, return_pred=False):
    """The run_probe_cell statistic evaluated on precomputed fold factors.
    With ``return_pred`` the result dict gains a ``pred`` key: the real
    fit's best-alpha pooled held-out predictions (used by the amendment-2
    secondary metrics; never fed back into any primary statistic)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    if return_pred:
        real, pred = _cv_pooled_best_factored(factors, y, task, return_pred=True)
    else:
        real = _cv_pooled_best_factored(factors, y, task)
    shuf_scores = [
        _cv_pooled_best_factored(factors, _shuffle_group_coherent(y, groups, rng), task)
        for _ in range(N_SHUFFLES)
    ]
    shuffled = float(np.mean(shuf_scores))
    shuffled_std = float(np.std(shuf_scores))
    floor = _floor(y, groups, task, splits=[(f["tr"], f["te"]) for f in factors])
    out = {
        "real": real,
        "shuffled": shuffled,
        "shuffled_std": shuffled_std,
        "floor": floor,
        "selectivity": real - shuffled,
        "n": len(y),
        "n_groups": len(np.unique(groups)),
    }
    if return_pred:
        out["pred"] = pred
    return out


def run_probe_cell(X, y, groups, task="reg", seed=0, return_pred=False):
    """One probe cell: real signal, group-coherent shuffled control, floor.

    Returns a dict with keys ``real, shuffled, shuffled_std, floor,
    selectivity, n, n_groups``. ``task="reg"``: ridge, metric R² (pooled
    held-out predictions). ``task="clf"``: logistic, metric balanced
    accuracy. ``return_pred=True`` adds a ``pred`` key holding the real
    fit's best-alpha pooled held-out predictions (secondary-metric input;
    the base keys are unchanged).

    ``shuffled`` is the mean over N_SHUFFLES independent group-coherent
    permutations, each with its own independent ALPHAS search (see module
    docstring for why the alpha is not shared with ``real``); ``shuffled_std``
    is that mean's sample std, surfaced so a small selectivity isn't read as
    "no signal" when it's actually within the null's own draw-to-draw noise.
    A single permutation's pooled score is high-variance when there are only
    a handful of groups (10-20, the regime this study runs in): which
    specific group-level values land in the held-out folds shifts the
    mean-predictor baseline by as much as the model's actual fit, so one draw
    can make ``selectivity`` look nonzero for a target with provably no
    signal. Averaging over several permutations is the standard fix for a
    small-sample permutation control.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    factors = _fold_factors(X, groups)
    return _cell_from_factors(factors, y, groups, task, seed, return_pred=return_pred)


def time_resolved(X, y, groups, step_rel, bins, task="reg", seed=0):
    """One ``run_probe_cell`` result per half-open ``[lo, hi)`` bin of
    ``step_rel``, each row tagged with ``bin_lo``/``bin_hi``."""
    step_rel = np.asarray(step_rel)
    rows = []
    for lo, hi in bins:
        mask = (step_rel >= lo) & (step_rel < hi)
        row = run_probe_cell(X[mask], y[mask], groups[mask], task=task, seed=seed)
        row = {**row, "bin_lo": lo, "bin_hi": hi}
        rows.append(row)
    return rows


def sweep(acts, targets, groups, masks, layers, positions, task="reg", seed=0, extra_metrics=None):
    """Full (target, layer, position, mask) grid of ``run_probe_cell`` calls
    over ``acts[:, layer, position, :]``. ``acts`` is upcast f16->f32 once.

    ``task`` is either one task string for every target (as before) or a
    dict ``{target_name: task}``. Split indices are computed once per mask
    and the per-fold SVD factors once per (layer, position, mask), shared
    across all targets and shuffle draws — cell statistics are unchanged
    (each cell still seeds its own rng with ``seed``).

    ``extra_metrics`` (amendment-2 secondaries): optional dict
    ``{target_name: callable(y_cell, pred_cell, row_idx) -> dict}`` — for
    those targets the real fit's pooled held-out predictions are captured
    and the callable's returned items become extra DataFrame columns (NaN
    for every other row). ``row_idx`` is the cell's original row indices.
    """
    acts = np.asarray(acts, dtype=np.float32)
    groups = np.asarray(groups)
    task_of = task if isinstance(task, dict) else {t: task for t in targets}
    extra_metrics = extra_metrics or {}
    masks = {m: np.asarray(v) for m, v in masks.items()}
    splits_by_mask = {m: _group_splits(np.zeros((v.sum(), 1)), groups[v]) for m, v in masks.items()}
    cells = {}
    for layer in layers:
        for position in positions:
            X_lp = acts[:, layer, position, :]
            for mname, mask in masks.items():
                factors = _fold_factors(X_lp[mask], groups[mask], splits=splits_by_mask[mname])
                for tname, y in targets.items():
                    y = np.asarray(y)
                    want_pred = tname in extra_metrics
                    cell = _cell_from_factors(
                        factors, y[mask], groups[mask], task_of[tname], seed,
                        return_pred=want_pred,
                    )
                    if want_pred:
                        pred = cell.pop("pred")
                        cell.update(extra_metrics[tname](y[mask], pred, np.flatnonzero(mask)))
                    cells[(tname, layer, position, mname)] = cell
    rows = [
        {"target": t, "layer": l, "position": p, "mask": m, **cells[(t, l, p, m)]}
        for t in targets
        for l in layers
        for p in positions
        for m in masks
    ]
    return pd.DataFrame(rows)

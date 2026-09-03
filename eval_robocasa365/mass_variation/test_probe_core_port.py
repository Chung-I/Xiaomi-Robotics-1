# Copyright (C) 2026 Xiaomi Corporation.
"""Smoke tests for the ported probe core (Plan-2 Task 2, Step 1).

The 4 behavioral tests below are PORTED from the origin suite of the copied
``analysis/probe_core.py`` (sibling pi0.5/RoboLab probing study,
``analysis/mass_com/test_probe_core.py`` in that worktree) -- only the
import path is adapted; fixtures, thresholds, and assertion semantics are
kept verbatim. They pin the grouped-CV + selectivity semantics the
certificates rely on:

1. real signal beats the group-coherent shuffled control and the floor;
2. a per-group-constant label with NO signal (pure group memorization) is
   exposed by selectivity ~ 0 -- the anti-memorization guarantee;
3. ``time_resolved`` bins tag their rows;
4. ``sweep`` produces the full (target, layer, position, mask) grid with
   the expected columns.
"""

from __future__ import annotations

import numpy as np

from eval_robocasa365.mass_variation.analysis.probe_core import (
    run_probe_cell,
    sweep,
    time_resolved,
)


def _grouped_data(n_groups=10, per=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * per, d)).astype(np.float32)
    groups = np.repeat(np.arange(n_groups), per)
    return X, groups, rng


def test_real_signal_beats_shuffled_and_floor():
    X, groups, rng = _grouped_data()
    w = rng.normal(size=X.shape[1])
    y = X @ w + 0.1 * rng.normal(size=len(X))
    r = run_probe_cell(X, y, groups)
    assert r["real"] > 0.8 and r["selectivity"] > 0.6
    assert r["shuffled"] < 0.2 and r["floor"] <= 0.0


def test_per_episode_constant_label_with_no_signal_is_caught_by_selectivity():
    # per-group-constant target, activations pure noise: real accuracy can sit
    # above the naive floor via group memorization; group-coherent shuffling
    # must expose it (selectivity ~ 0)
    X, groups, rng = _grouped_data()
    y = np.repeat(rng.normal(size=10), 40)  # per-episode constant, no signal
    r = run_probe_cell(X, y, groups)
    assert abs(r["selectivity"]) < 0.15


def test_time_resolved_bins_and_tags():
    X, groups, rng = _grouped_data()
    step_rel = np.tile(np.arange(-10, 30), 10)
    y = (step_rel > 0) * 1.0 + 0.05 * rng.normal(size=len(X))  # decodable only by phase
    rows = time_resolved(X, y, groups, step_rel, bins=[(-10, 0), (0, 15), (15, 30)], task="reg")
    assert len(rows) == 3 and all("bin_lo" in r for r in rows)


def test_sweep_grid_shape():
    X, groups, rng = _grouped_data(per=20)
    acts = np.stack([np.stack([X, X * 0.5], axis=1)] * 3, axis=1)  # (N, 3, 2, d)
    targets = {"m": X @ rng.normal(size=X.shape[1])}
    masks = {"all": np.ones(len(X), bool), "half": np.arange(len(X)) % 2 == 0}
    df = sweep(acts.astype(np.float16), targets, groups, masks, layers=[0, 2], positions=[0, 1])
    assert len(df) == 1 * 2 * 2 * 2  # targets x layers x positions x masks
    assert set(df.columns) >= {
        "target", "layer", "position", "mask", "real", "shuffled", "shuffled_std", "selectivity", "floor", "n",
    }

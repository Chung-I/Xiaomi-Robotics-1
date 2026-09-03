# Copyright (C) 2026 Xiaomi Corporation.
"""Unit tests for the pure helpers of the Plan-2 Task 2 certificates CLI.

All pure -- no sim, no GPU, no corpus. The end-to-end certificate run is
exercised live by the CLI itself (its output is committed as
``output/mass_variation/analysis/certificates.json``).
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_robocasa365.mass_variation.analysis.certificates import (
    DEGENERATE_VAR_TOL,
    certificate_cell,
    degenerate_cell,
    fit_k_eff,
    per_episode_windows,
    rank_acc_levels,
)


def _rank_acc_naive(y_true, y_pred):
    """O(n^2) reference: fraction of different-true-value pairs ordered
    correctly by the prediction; a tied prediction counts as incorrect."""
    correct = total = 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1
            if np.sign(y_pred[i] - y_pred[j]) == np.sign(y_true[i] - y_true[j]):
                correct += 1
    return correct / total if total else float("nan")


class TestRankAccLevels:
    def test_matches_naive_on_random_three_level_data(self):
        rng = np.random.default_rng(0)
        y_true = rng.choice([-1.386, 0.0, 0.693], size=60)
        y_pred = y_true + rng.normal(scale=1.0, size=60)
        assert rank_acc_levels(y_true, y_pred) == pytest.approx(
            _rank_acc_naive(y_true, y_pred)
        )

    def test_perfect_and_inverted(self):
        y_true = np.repeat([0.0, 1.0, 2.0], 5)
        assert rank_acc_levels(y_true, y_true.copy()) == 1.0
        assert rank_acc_levels(y_true, -y_true) == 0.0

    def test_tied_predictions_count_as_incorrect(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 0.5])
        assert rank_acc_levels(y_true, y_pred) == 0.0

    def test_no_valid_pairs_is_nan(self):
        y = np.zeros(4)
        assert np.isnan(rank_acc_levels(y, np.arange(4.0)))


class TestPerEpisodeWindows:
    def test_windows_never_cross_episode_boundaries(self):
        # Two episodes with wildly different constant values: if a window
        # crossed the boundary, episode B's step-0 window would contain
        # episode A's values instead of B's left-clamped own first frame.
        x = np.concatenate([np.full((5, 2), 1.0), np.full((4, 2), 9.0)])
        eid = np.array(["a"] * 5 + ["b"] * 4)
        W = per_episode_windows(x, eid, k=3, stride=1)
        assert W.shape == (9, 6)
        assert np.all(W[5] == 9.0)  # b's first step: left-clamp on b itself

    def test_matches_window_stack_semantics_within_episode(self):
        from eval_robocasa365.mass_variation.analysis.dataset import window_stack

        rng = np.random.default_rng(1)
        x = rng.normal(size=(7, 3))
        eid = np.array(["only"] * 7)
        W = per_episode_windows(x, eid, k=4, stride=2)
        ref = window_stack(x, k=4, stride=2).reshape(7, -1)
        np.testing.assert_array_equal(W, ref)

    def test_noncontiguous_episode_rows_raise(self):
        x = np.zeros((4, 1))
        eid = np.array(["a", "b", "a", "b"])
        with pytest.raises(ValueError, match="contiguous"):
            per_episode_windows(x, eid, k=2, stride=1)


class TestCertificateCell:
    def _data(self, n_seeds=10, eps_per_seed=3, per=20, d=8, seed=0):
        """Synthetic corpus shaped like the study: (condition x seed)
        episodes, cv groups = seed, shuffle groups = episode, target
        constant per episode (3 'mass levels' tied to the condition)."""
        rng = np.random.default_rng(seed)
        levels = np.array([-1.386, 0.0, 0.693])
        X, y, cv_g, sh_g = [], [], [], []
        for cond in range(eps_per_seed):
            for s in range(n_seeds):
                X.append(rng.normal(size=(per, d)))
                y.append(np.full(per, levels[cond]))
                cv_g.append(np.full(per, s))
                sh_g.append(np.full(per, cond * n_seeds + s))
        return (np.concatenate(X), np.concatenate(y),
                np.concatenate(cv_g), np.concatenate(sh_g), rng)

    def test_real_signal_recovered(self):
        X, y, cv_g, sh_g, rng = self._data()
        X[:, 0] = y + 0.1 * rng.normal(size=len(y))  # plant the signal
        r = certificate_cell(X, y, cv_g, sh_g)
        assert r["r2_pooled"] > 0.8 and r["selectivity"] > 0.6
        assert len(r["r2_folds"]) == 5
        assert r["rank_acc"] > 0.9
        assert not r["degenerate"]

    def test_pure_noise_has_no_r2_and_no_selectivity(self):
        X, y, cv_g, sh_g, _ = self._data()
        r = certificate_cell(X, y, cv_g, sh_g)
        assert r["r2_pooled"] < 0.1
        assert abs(r["selectivity"]) < 0.15

    def test_episode_level_shuffle_not_near_noop(self):
        # Regression guard for the hybrid-group design choice: with cv
        # groups (seed) also used as SHUFFLE groups, probe_core's
        # block-swap branch would copy each seed's condition-ordered label
        # block onto another seed's -- a near no-op on this corpus layout
        # (same condition order everywhere), deflating selectivity to ~0
        # even for real signal. Episode-level shuffle groups keep the
        # control an actual label permutation.
        X, y, cv_g, sh_g, rng = self._data()
        X[:, 0] = y + 0.1 * rng.normal(size=len(y))
        r = certificate_cell(X, y, cv_g, sh_g)
        assert r["shuffled"] < 0.2  # a near-no-op shuffle would score ~r2_pooled

    def test_degenerate_guard(self):
        cell = degenerate_cell("empty mask")
        assert cell["degenerate"] and cell["r2_pooled"] is None
        X, y, cv_g, sh_g, _ = self._data()
        r = certificate_cell(X, np.zeros_like(y), cv_g, sh_g)
        assert r["degenerate"] and r["r2_pooled"] is None
        assert np.var(np.zeros_like(y)) < DEGENERATE_VAR_TOL


class TestFitKEff:
    def test_recovers_planted_slope(self):
        rng = np.random.default_rng(2)
        deficit = rng.normal(size=500)
        fz = 3.5 * deficit - 1.0 + 0.01 * rng.normal(size=500)
        cond = np.array(["MassLight", "MassMedium", "MassHeavy"] * 167)[:500]
        out = fit_k_eff(fz, deficit, cond)
        assert out["slope"] == pytest.approx(3.5, abs=0.01)
        assert out["intercept"] == pytest.approx(-1.0, abs=0.01)
        assert out["r2"] > 0.99
        assert set(out["per_condition"]) == {"MassLight", "MassMedium", "MassHeavy"}
        assert out["n"] == 500

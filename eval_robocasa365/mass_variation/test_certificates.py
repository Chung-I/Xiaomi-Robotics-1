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
    K_DEFICIT,
    K_POLICY,
    K_RAW_FT,
    axis_angle_to_matrix,
    build_features,
    certificate_cell,
    degenerate_cell,
    derived_force_features,
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


class TestAxisAngleToMatrix:
    def test_quarter_turn_about_z_maps_x_to_y(self):
        R = axis_angle_to_matrix(np.array([[0.0, 0.0, np.pi / 2]]))[0]
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(R @ [0, 1, 0], [-1, 0, 0], atol=1e-12)

    def test_zero_vector_is_identity_and_batch_shape(self):
        aa = np.zeros((4, 3))
        aa[1] = [np.pi, 0, 0]
        R = axis_angle_to_matrix(aa)
        assert R.shape == (4, 3, 3)
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-12)
        # rotations are orthonormal: R R^T = I
        np.testing.assert_allclose(R[1] @ R[1].T, np.eye(3), atol=1e-12)


class TestDerivedForceFeatures:
    def test_norms_and_known_rotation(self):
        F = np.array([[1.0, 2.0, 3.0]])
        tau = np.array([[0.0, 4.0, 0.0]])
        ps = np.zeros((1, 14), dtype=np.float32)
        ps[0, 3:6] = [0.0, 0.0, np.pi / 2]  # EEF rel base: +90 deg about z
        out = derived_force_features(F, tau, ps)
        assert out.shape == (1, 8)
        assert out[0, 0] == pytest.approx(np.sqrt(14), rel=1e-6)  # |F|
        assert out[0, 1] == pytest.approx(4.0, rel=1e-6)  # |tau|
        # base rot identity => F_world == F_base == Rz(90) @ F = (-2, 1, 3)
        np.testing.assert_allclose(out[0, 2:5], [-2, 1, 3], atol=1e-6)
        np.testing.assert_allclose(out[0, 5:8], [-2, 1, 3], atol=1e-6)

    def test_base_yaw_moves_world_not_base(self):
        F = np.array([[1.0, 0.0, 0.0]])
        tau = np.zeros((1, 3))
        ps = np.zeros((1, 14), dtype=np.float32)
        ps[0, 11:14] = [0.0, 0.0, np.pi / 2]  # base yaw +90 deg
        out = derived_force_features(F, tau, ps)
        np.testing.assert_allclose(out[0, 5:8], [1, 0, 0], atol=1e-6)  # F_base
        np.testing.assert_allclose(out[0, 2:5], [0, 1, 0], atol=1e-6)  # F_world


class TestBuildFeatures:
    """Amendment B MINOR review item: column counts + no-contamination."""

    def _fake_study(self, n=10, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "episode_id": np.array(["ep_a"] * (n // 2) + ["ep_b"] * (n - n // 2)),
            "ee_force": rng.normal(size=(n, 3)).astype(np.float32),
            "ee_torque": rng.normal(size=(n, 3)).astype(np.float32),
            "commanded_delta": rng.normal(size=(n, 6)).astype(np.float32),
            "achieved_eef_delta": rng.normal(size=(n, 6)).astype(np.float32),
            "policy_state_14": rng.normal(size=(n, 14)).astype(np.float32),
            "policy_state_16": rng.normal(size=(n, 16)).astype(np.float32),
        }

    def test_column_counts(self):
        d = self._fake_study()
        assert build_features("raw_ft", d).shape[1] == K_RAW_FT * 14  # 6 raw + 8 derived
        assert build_features("policy_obs_xr1", d).shape[1] == K_POLICY * 14
        assert build_features("policy_obs_pi05", d).shape[1] == 16
        assert build_features("deficit", d).shape[1] == K_DEFICIT * 12

    def test_no_contamination(self):
        d = self._fake_study()
        base = {c: build_features(c, d) for c in
                ("raw_ft", "policy_obs_xr1", "policy_obs_pi05", "deficit")}

        # policy_obs certs must ignore every force channel (no circularity)
        d2 = self._fake_study()
        d2["ee_force"] = d2["ee_force"] + 100.0
        d2["ee_torque"] = d2["ee_torque"] + 100.0
        np.testing.assert_array_equal(
            build_features("policy_obs_xr1", d2), base["policy_obs_xr1"])
        np.testing.assert_array_equal(
            build_features("policy_obs_pi05", d2), base["policy_obs_pi05"])
        np.testing.assert_array_equal(build_features("deficit", d2), base["deficit"])
        assert not np.array_equal(build_features("raw_ft", d2), base["raw_ft"])

        # raw_ft may use ONLY policy_state_14's ORIENTATION dims (3:6, 11:14)
        # -- perturbing position/gripper dims must not change it; and it must
        # ignore policy_state_16 and the delta channels entirely
        d3 = self._fake_study()
        d3["policy_state_14"][:, 0:3] += 5.0
        d3["policy_state_14"][:, 6:11] += 5.0
        d3["policy_state_16"] += 5.0
        d3["commanded_delta"] += 5.0
        d3["achieved_eef_delta"] += 5.0
        np.testing.assert_array_equal(build_features("raw_ft", d3), base["raw_ft"])

        # ... while orientation dims DO enter (documented amendment-B use)
        d4 = self._fake_study()
        d4["policy_state_14"][:, 3:6] += 1.0
        assert not np.array_equal(build_features("raw_ft", d4), base["raw_ft"])


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

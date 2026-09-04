# Copyright (C) 2026 Xiaomi Corporation.
"""Pure unit tests for Plan-2 amendment C (the vision certificate).

Two things are TDD'd here, per the amendment's own risk list:

1. **Visual-format assembly** -- which frames each model's format reads.
   XR1's 4-frame stride-2 window must follow ``entry.sample_history``'s
   index rule EXACTLY (the reference is imported and driven with a real
   ``collections.deque``, exactly as ``entry_mass.run_episode`` drives it,
   rather than restated); pi0.5's format is the single current frame.
2. **PCA-in-fold wiring** -- the amendment's explicit leakage rule: the
   PCA must be fitted on the TRAINING rows of each fold only. The tests
   assert the fitted mean/components are functions of the train block
   alone (test-fold rows can be arbitrarily corrupted without moving
   them) and that they are NOT the all-rows fit.

Plus the frame keep-set rule and the grayscale thumbnail, which decide what
``render_frames.py`` has to store.

All pure: no sim, no GPU, no torch, no disk corpus.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from eval_robocasa365.entry import sample_history
from eval_robocasa365.mass_variation import render_frames
from eval_robocasa365.mass_variation.analysis import vision_certificate as vc


# ---------------------------------------------------------------------------
# frame keep-set (render_frames.frame_steps_needed)
# ---------------------------------------------------------------------------


def _masks(precontact, carry):
    return {
        "precontact": np.asarray(precontact, dtype=bool),
        "carry": np.asarray(carry, dtype=bool),
    }


class TestFrameStepsNeeded:
    def test_precontact_steps_are_kept_verbatim(self):
        m = _masks([1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(render_frames.frame_steps_needed(m), [0, 1, 2])

    def test_carry_step_pulls_its_whole_window(self):
        # only step 9 is carry -> needs 9, 7, 5, 3
        m = _masks([0] * 10, [0] * 9 + [1])
        np.testing.assert_array_equal(render_frames.frame_steps_needed(m), [3, 5, 7, 9])

    def test_carry_window_left_clamps_at_zero(self):
        # carry at step 1 -> 1, max(0,-1), max(0,-3), max(0,-5) = {0, 1}
        m = _masks([0, 0, 0], [0, 1, 0])
        np.testing.assert_array_equal(render_frames.frame_steps_needed(m), [0, 1])

    def test_union_is_sorted_unique(self):
        m = _masks([1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1])
        got = render_frames.frame_steps_needed(m)
        assert got.tolist() == sorted(set(got.tolist()))
        # carry {6,7} -> {6,4,2,0} u {7,5,3,1}; precontact {0,1,2,3}
        assert got.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_empty_masks_give_empty_keep_set(self):
        m = _masks([0, 0, 0], [0, 0, 0])
        assert render_frames.frame_steps_needed(m).shape == (0,)

    def test_keep_set_never_exceeds_episode_length(self):
        m = _masks([1, 0, 0, 0, 0], [0, 0, 0, 1, 1])
        got = render_frames.frame_steps_needed(m)
        assert got.min() >= 0 and got.max() < 5

    def test_rejects_mismatched_masks(self):
        with pytest.raises(ValueError):
            render_frames.frame_steps_needed(_masks([1, 0], [0, 0, 0]))


class TestGrayscaleThumb:
    def test_shape_and_dtype(self):
        rgb = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        out = render_frames.to_grayscale_thumb(rgb, 96)
        assert out.shape == (96, 96) and out.dtype == np.uint8

    def test_deterministic(self):
        rgb = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        a = render_frames.to_grayscale_thumb(rgb, 96)
        b = render_frames.to_grayscale_thumb(rgb, 96)
        np.testing.assert_array_equal(a, b)

    def test_constant_frame_stays_constant(self):
        rgb = np.full((256, 256, 3), 137, dtype=np.uint8)
        out = render_frames.to_grayscale_thumb(rgb, 96)
        assert out.min() == out.max() == 137

    def test_rejects_non_rgb(self):
        with pytest.raises(ValueError):
            render_frames.to_grayscale_thumb(np.zeros((256, 256), dtype=np.uint8))


# ---------------------------------------------------------------------------
# visual-format assembly
# ---------------------------------------------------------------------------


def _reference_window_steps(t: int, k: int, stride: int) -> list[int]:
    """``entry.sample_history``'s answer, obtained by DRIVING it -- a live
    bounded deque fed step ids 0..t, exactly like ``entry_mass.run_episode``
    feeds it observations (``queue_length = (k - 1) * stride + 1``)."""
    queue = collections.deque(maxlen=(k - 1) * stride + 1)
    for step in range(t + 1):
        queue.append(np.array([step]))
    return [int(v[0]) for v in sample_history(queue, k, stride)]


class TestWindowSteps:
    @pytest.mark.parametrize("t", list(range(0, 25)))
    def test_matches_entry_sample_history(self, t):
        assert vc.window_steps(t, k=4, stride=2) == _reference_window_steps(t, 4, 2)

    def test_xr1_format_is_four_frames_stride_two(self):
        assert vc.window_steps(20, **vc.VISUAL_FORMATS["xr1"]["window"]) == [14, 16, 18, 20]

    def test_pi05_format_is_the_single_current_frame(self):
        assert vc.window_steps(20, **vc.VISUAL_FORMATS["pi05_robocasa"]["window"]) == [20]

    def test_newest_frame_is_last(self):
        assert vc.window_steps(9, k=4, stride=2)[-1] == 9

    def test_left_clamp_repeats_step_zero(self):
        assert vc.window_steps(1, k=4, stride=2) == [0, 0, 0, 1]


class TestGatherVisualRows:
    def _frames(self, n_steps, n_cam=3, side=4):
        # frame value encodes (step, camera) so gathers are checkable
        f = np.zeros((n_steps, n_cam, side, side), dtype=np.uint8)
        for s in range(n_steps):
            for c in range(n_cam):
                f[s, c] = (s * 10 + c) % 256
        return f

    def test_pi05_gathers_one_frame_per_row(self):
        frames = self._frames(6)
        steps = np.arange(6)
        out = vc.gather_visual_rows(frames, steps, np.array([2, 5]), k=1, stride=1)
        assert out.shape == (2, 1, 3, 4, 4)
        assert out[0, 0, 0, 0, 0] == 20 and out[1, 0, 2, 0, 0] == 52

    def test_xr1_gathers_the_sample_history_window(self):
        frames = self._frames(12)
        steps = np.arange(12)
        out = vc.gather_visual_rows(frames, steps, np.array([9]), k=4, stride=2)
        assert out.shape == (1, 4, 3, 4, 4)
        got = [int(out[0, i, 0, 0, 0]) // 10 for i in range(4)]
        assert got == _reference_window_steps(9, 4, 2)

    def test_works_on_a_sparse_stored_keep_set(self):
        # only steps {3,5,7,9} stored, as render_frames would store for a
        # single carry step at 9
        keep = np.array([3, 5, 7, 9])
        frames = self._frames(12)[keep]
        out = vc.gather_visual_rows(frames, keep, np.array([9]), k=4, stride=2)
        got = [int(out[0, i, 0, 0, 0]) // 10 for i in range(4)]
        assert got == [3, 5, 7, 9]

    def test_missing_stored_step_raises(self):
        keep = np.array([5, 7, 9])  # step 3 deliberately absent
        frames = self._frames(12)[keep]
        with pytest.raises(KeyError):
            vc.gather_visual_rows(frames, keep, np.array([9]), k=4, stride=2)


class TestModelFormatTransform:
    def test_pi05_resize_with_pad_is_identity_at_storage_size(self):
        # square input already at the target side -> openpi's resize_with_pad
        # is a no-op, so pi0.5's own format adds nothing at 96x96
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 256, (96, 96), dtype=np.uint8)
        np.testing.assert_array_equal(vc.apply_format_transform(frame, "pi05_robocasa"), frame)

    def test_xr1_applies_its_own_center_crop(self):
        rng = np.random.default_rng(4)
        frame = rng.integers(0, 256, (96, 96), dtype=np.uint8)
        out = vc.apply_format_transform(frame, "xr1")
        assert out.shape == frame.shape and out.dtype == np.uint8
        assert not np.array_equal(out, frame)  # 0.95 crop is a real zoom

    def test_xr1_crop_matches_entry_center_crop(self):
        from eval_robocasa365.entry import center_crop
        rng = np.random.default_rng(5)
        frame = rng.integers(0, 256, (96, 96), dtype=np.uint8)
        expected = np.asarray(center_crop(frame, vc.XR1_CROP_RATIO), dtype=np.uint8)
        np.testing.assert_array_equal(vc.apply_format_transform(frame, "xr1"), expected)

    def test_constant_frame_is_preserved_by_both_transforms(self):
        frame = np.full((96, 96), 200, dtype=np.uint8)
        for model in ("xr1", "pi05_robocasa"):
            out = vc.apply_format_transform(frame, model)
            assert out.min() == out.max() == 200


# ---------------------------------------------------------------------------
# PCA-in-fold wiring (the amendment's leakage rule)
# ---------------------------------------------------------------------------


def _low_rank(n, d, rank, rng):
    return (rng.standard_normal((n, rank)) @ rng.standard_normal((rank, d))).astype(np.float32)


class TestPCAFit:
    def test_recovers_a_low_rank_subspace(self):
        rng = np.random.default_rng(0)
        X = _low_rank(40, 30, 3, rng)
        mean, comps = vc.pca_fit(X, n_components=3, seed=0)
        Z = vc.pca_transform(X, mean, comps)
        recon = Z @ comps + mean
        assert np.allclose(recon, X, atol=1e-2)

    def test_components_are_orthonormal(self):
        rng = np.random.default_rng(1)
        X = _low_rank(40, 30, 8, rng)
        _, comps = vc.pca_fit(X, n_components=5, seed=0)
        np.testing.assert_allclose(comps @ comps.T, np.eye(5), atol=1e-4)

    def test_caps_components_at_the_available_rank(self):
        rng = np.random.default_rng(2)
        X = _low_rank(12, 40, 4, rng)
        _, comps = vc.pca_fit(X, n_components=512, seed=0)
        assert comps.shape[0] <= min(X.shape)

    def test_mean_is_the_row_mean(self):
        rng = np.random.default_rng(3)
        X = _low_rank(20, 15, 5, rng)
        mean, _ = vc.pca_fit(X, n_components=4, seed=0)
        np.testing.assert_allclose(mean, X.mean(axis=0), atol=1e-5)

    def test_deterministic_for_a_fixed_seed(self):
        rng = np.random.default_rng(4)
        X = _low_rank(30, 25, 6, rng)
        a = vc.pca_fit(X, n_components=4, seed=0)
        b = vc.pca_fit(X, n_components=4, seed=0)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class TestPCAInFold:
    def _setup(self, n_groups=10, per_group=6, d=20, rank=5, seed=0):
        rng = np.random.default_rng(seed)
        X = _low_rank(n_groups * per_group, d, rank, rng)
        groups = np.repeat(np.arange(n_groups), per_group)
        return X, groups

    def test_fold_pca_is_fitted_on_train_rows_only(self):
        X, groups = self._setup()
        factors = vc.pca_fold_factors(X, groups, n_components=4, seed=0)
        for f in factors:
            np.testing.assert_allclose(f["pca_mean"], X[f["tr"]].mean(axis=0), atol=1e-5)
            expected_mean, expected_comps = vc.pca_fit(X[f["tr"]], 4, seed=0)
            np.testing.assert_allclose(f["pca_components"], expected_comps, atol=1e-5)

    def test_fold_pca_is_not_the_all_rows_fit(self):
        # the leakage this rule exists to prevent: an all-rows PCA would put
        # the SAME mean/components in every fold
        X, groups = self._setup()
        all_mean, _ = vc.pca_fit(X, 4, seed=0)
        factors = vc.pca_fold_factors(X, groups, n_components=4, seed=0)
        for f in factors:
            assert not np.allclose(f["pca_mean"], all_mean, atol=1e-6)

    def test_corrupting_test_rows_cannot_move_the_fitted_pca(self):
        # the sharpest leakage probe: blow up ONLY the held-out rows of one
        # fold; a within-fold PCA is bit-identical, a leaked one is not
        X, groups = self._setup()
        base = vc.pca_fold_factors(X, groups, n_components=4, seed=0)
        X2 = X.copy()
        X2[base[0]["te"]] += 1e3
        poisoned = vc.pca_fold_factors(X2, groups, n_components=4, seed=0)
        np.testing.assert_array_equal(base[0]["pca_mean"], poisoned[0]["pca_mean"])
        np.testing.assert_array_equal(base[0]["pca_components"], poisoned[0]["pca_components"])

    def test_test_block_is_projected_with_the_train_fit(self):
        X, groups = self._setup()
        factors = vc.pca_fold_factors(X, groups, n_components=4, seed=0)
        for f in factors:
            Z_te = vc.pca_transform(X[f["te"]], f["pca_mean"], f["pca_components"])
            # probe_core's own factorisation of the PCA scores: G = (Z_te -
            # mu_Z) V, and Z_te is recoverable as G @ V^T + mu_Z
            assert f["G"].shape == (len(f["te"]), min(f["s"].shape[0], Z_te.shape[1]))

    def test_factors_carry_the_probe_core_keys(self):
        X, groups = self._setup()
        for f in vc.pca_fold_factors(X, groups, n_components=4, seed=0):
            assert {"tr", "te", "U", "s", "G"} <= set(f)
            assert f["U"].shape[0] == len(f["tr"])
            assert f["G"].shape[0] == len(f["te"])

    def test_folds_are_group_disjoint(self):
        X, groups = self._setup()
        for f in vc.pca_fold_factors(X, groups, n_components=4, seed=0):
            assert not (set(groups[f["tr"]]) & set(groups[f["te"]]))

    def test_recovers_a_group_level_signal_through_the_pca(self):
        # end-to-end sanity: a target that IS in the top PCA directions must
        # come back with a high R2 through the in-fold pipeline
        rng = np.random.default_rng(7)
        n_groups, per_group, d = 15, 8, 30
        y_group = rng.standard_normal(n_groups)
        groups = np.repeat(np.arange(n_groups), per_group)
        y = y_group[groups]
        direction = rng.standard_normal(d)
        X = (y[:, None] * direction[None, :] * 5.0
             + 0.1 * rng.standard_normal((n_groups * per_group, d))).astype(np.float32)
        cell = vc.vision_certificate_cell(X, y, groups, groups, n_components=4, seed=0)
        assert cell["r2_pooled"] > 0.8
        assert cell["selectivity"] > 0.5

    def test_degenerate_guards_match_the_certificates_module(self):
        empty = vc.vision_certificate_cell(
            np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0), np.zeros(0),
            n_components=2, seed=0)
        assert empty["degenerate"] and empty["r2_pooled"] is None
        const = vc.vision_certificate_cell(
            np.ones((20, 4), np.float32), np.zeros(20), np.repeat(np.arange(5), 4),
            np.repeat(np.arange(5), 4), n_components=2, seed=0)
        assert const["degenerate"]


class TestCellSchema:
    def test_matches_the_certificates_module_schema(self):
        from eval_robocasa365.mass_variation.analysis import certificates
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(10), 6)
        X = rng.standard_normal((60, 8)).astype(np.float32)
        y = rng.standard_normal(10)[groups]
        ridge_keys = set(certificates.certificate_cell(X, y, groups, groups))
        vision_keys = set(vc.vision_certificate_cell(X, y, groups, groups, n_components=4, seed=0))
        assert ridge_keys <= vision_keys

    def test_required_json_keys_present_after_tagging(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(10), 6)
        X = rng.standard_normal((60, 8)).astype(np.float32)
        y = rng.standard_normal(10)[groups]
        cell = vc.vision_certificate_cell(X, y, groups, groups, n_components=4, seed=0)
        cell.update(model="xr1", certificate="policy_obs_vision", kind="ridge_pca",
                    mask="carry", input_channels=["x"])
        for key in ("model", "certificate", "kind", "mask", "r2_pooled", "r2_folds",
                    "rank_acc", "selectivity", "shuffled", "shuffled_std", "floor",
                    "n", "n_groups", "input_channels", "degenerate"):
            assert key in cell, key

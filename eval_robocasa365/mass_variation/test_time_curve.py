# Copyright (C) 2026 Xiaomi Corporation.
"""Pure-helper tests for the EXPLORATORY time-curve analysis
(``analysis/time_curve.py``).

TDD scope: the contact-relative binning seam only (everything here runs
without acts npz, GPU, or sim). A bug here would shift every point on the
figure's x axis, so it is pinned before anything is plotted:

- ``first_contact_steps``: the per-episode first ``grasped==True`` step; an
  episode that never grasps is ABSENT from the table (never silently 0).
- ``contact_relative_steps``: ``step - first_contact`` per row, NaN for a
  row whose episode never grasped -- so those rows can never land in a bin.
- ``make_bins``: the half-open ``[lo, hi)`` step-bin grid, and its refusal
  of a span that is not a whole number of bins.
- ``assign_bins``: row -> bin index, ``-1`` for NaN / out-of-range, exact
  edge behaviour (lower edge inclusive, upper edge exclusive), and the
  empty-input case.
- ``place_step`` / ``place_steps`` / ``median_place_s``: the place/release
  event the figure's third vertical rule marks. Its failure modes are the
  ones that would silently move that rule: an episode that never releases,
  one that releases and RE-grasps (the first release is the place), one
  that never lifted at all, and a release recorded while the object is
  still on its way up.
- ``window_available``: which captured rows the camera-pixel reader may be
  scored on -- the render keep-set covers ``precontact`` and ``carry``
  windows only, so a row whose 4-frame stride-2 window is not fully stored
  must be EXCLUDED, never silently substituted with a neighbouring frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_robocasa365.mass_variation.analysis.time_curve import (
    assign_bins,
    contact_relative_steps,
    first_contact_steps,
    make_bins,
    median_place_s,
    place_step,
    place_steps,
    window_available,
)


# ---------------------------------------------------------------------------
# first_contact_steps
# ---------------------------------------------------------------------------


def test_first_contact_steps_picks_the_first_true_per_episode():
    eid = np.array(["a", "a", "a", "b", "b", "b"])
    step = np.array([0, 1, 2, 0, 1, 2])
    grasped = np.array([False, True, True, False, False, True])
    assert first_contact_steps(eid, step, grasped) == {"a": 1, "b": 2}


def test_first_contact_steps_omits_never_grasped_episodes():
    eid = np.array(["a", "a", "b", "b"])
    step = np.array([0, 1, 0, 1])
    grasped = np.array([False, False, True, True])
    table = first_contact_steps(eid, step, grasped)
    assert table == {"b": 0}
    assert "a" not in table  # absent, never 0


def test_first_contact_steps_uses_the_step_column_not_the_row_offset():
    # rows of episode "a" start at step 100 (a subset / offset corpus)
    eid = np.array(["a", "a", "a"])
    step = np.array([100, 101, 102])
    grasped = np.array([False, True, False])
    assert first_contact_steps(eid, step, grasped) == {"a": 101}


def test_first_contact_steps_empty_input():
    empty = np.array([], dtype=object)
    assert first_contact_steps(empty, np.array([]), np.array([], dtype=bool)) == {}


# ---------------------------------------------------------------------------
# contact_relative_steps
# ---------------------------------------------------------------------------


def test_contact_relative_steps_centres_each_episode_on_its_own_contact():
    eid = np.array(["a", "a", "a", "b", "b"])
    step = np.array([0, 1, 2, 5, 6])
    table = {"a": 1, "b": 6}
    rel = contact_relative_steps(eid, step, table)
    np.testing.assert_allclose(rel, [-1.0, 0.0, 1.0, -1.0, 0.0])


def test_contact_relative_steps_is_nan_for_an_episode_with_no_contact():
    eid = np.array(["a", "a", "b"])
    step = np.array([0, 1, 0])
    rel = contact_relative_steps(eid, step, {"a": 1})
    np.testing.assert_allclose(rel[:2], [-1.0, 0.0])
    assert np.isnan(rel[2])


def test_contact_relative_steps_empty_input():
    rel = contact_relative_steps(np.array([], dtype=object), np.array([]), {})
    assert rel.shape == (0,)


# ---------------------------------------------------------------------------
# make_bins
# ---------------------------------------------------------------------------


def test_make_bins_is_a_contiguous_half_open_grid():
    assert make_bins(-64, 64, 32) == ((-64, -32), (-32, 0), (0, 32), (32, 64))


def test_make_bins_places_a_bin_edge_exactly_at_contact():
    edges = {lo for lo, _ in make_bins(-128, 192, 32)}
    assert 0 in edges  # zero must be a bin BOUNDARY, never mid-bin


def test_make_bins_rejects_a_span_that_is_not_a_whole_number_of_bins():
    with pytest.raises(ValueError):
        make_bins(-64, 50, 32)


def test_make_bins_rejects_a_non_positive_width():
    with pytest.raises(ValueError):
        make_bins(-64, 64, 0)


# ---------------------------------------------------------------------------
# assign_bins
# ---------------------------------------------------------------------------


def test_assign_bins_lower_edge_inclusive_upper_edge_exclusive():
    bins = make_bins(-64, 64, 32)  # [-64,-32) [-32,0) [0,32) [32,64)
    rel = np.array([-64.0, -33.0, -32.0, -1.0, 0.0, 31.0, 32.0, 63.0])
    np.testing.assert_array_equal(assign_bins(rel, bins), [0, 0, 1, 1, 2, 2, 3, 3])


def test_assign_bins_marks_out_of_range_and_nan_as_minus_one():
    bins = make_bins(-64, 64, 32)
    rel = np.array([-65.0, 64.0, 1000.0, np.nan])
    np.testing.assert_array_equal(assign_bins(rel, bins), [-1, -1, -1, -1])


def test_assign_bins_empty_input():
    out = assign_bins(np.array([]), make_bins(-64, 64, 32))
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_assign_bins_with_no_bins_returns_all_minus_one():
    np.testing.assert_array_equal(assign_bins(np.array([0.0, 5.0]), ()), [-1, -1])


# ---------------------------------------------------------------------------
# place_step -- the place/release event (figure rule 3)
# ---------------------------------------------------------------------------


def test_place_step_is_the_first_release_at_a_non_rising_height():
    #      step:      0    1    2    3    4    5    6
    grasped = np.array([0, 1, 1, 1, 1, 0, 0], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.1, 1.3, 1.5, 1.5, 1.5])
    # liftoff at step 2; the gripper opens at step 5 with the object resting
    assert place_step(grasped, 2, obj_z) == 5


def test_place_step_returns_none_when_the_object_is_never_released():
    grasped = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.2, 1.4, 1.5, 1.5])
    assert place_step(grasped, 2, obj_z) is None


def test_place_step_takes_the_FIRST_place_when_the_object_is_regrasped():
    # released at 4 (resting), picked up again at 6, released again at 9
    grasped = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.2, 1.4, 1.4, 1.4, 1.4, 1.6, 1.8, 1.8])
    assert place_step(grasped, 2, obj_z) == 4


def test_place_step_returns_none_for_a_never_lifted_episode():
    # recorder.liftoff_step's -1 convention: grasped, but never lifted
    grasped = np.array([0, 1, 1, 0, 0], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert place_step(grasped, -1, obj_z) is None


def test_place_step_skips_a_release_while_the_object_is_still_rising():
    # gripper reads open at step 3 but the object is still going up (a
    # transient contact-flag drop mid-lift, not a place); the real place is 5
    grasped = np.array([0, 1, 1, 0, 1, 0], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.1, 1.3, 1.5, 1.5])
    assert place_step(grasped, 2, obj_z) == 5


def test_place_step_ignores_ungrasped_steps_before_liftoff():
    # steps 0-1 are pre-contact (ungrasped, object at rest); the place is 5
    grasped = np.array([0, 0, 1, 1, 1, 0], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.0, 1.2, 1.4, 1.4])
    assert place_step(grasped, 3, obj_z) == 5


def test_place_step_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        place_step(np.array([True, False]), 0, np.array([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# place_steps / median_place_s
# ---------------------------------------------------------------------------


def _two_episode_ds():
    """Two episodes: 'a' places, 'b' never releases. 'a' contacts at step 1,
    lifts (first carry step) at 2, places at 5 -> +4 steps after contact."""
    eid = np.array(["a"] * 7 + ["b"] * 6)
    step = np.concatenate([np.arange(7), np.arange(6)])
    grasped = np.array([0, 1, 1, 1, 1, 0, 0] + [0, 1, 1, 1, 1, 1], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.1, 1.3, 1.5, 1.5, 1.5]
                     + [1.0, 1.0, 1.2, 1.4, 1.5, 1.5])
    carry = np.array([0, 0, 1, 1, 1, 0, 0] + [0, 0, 1, 1, 1, 1], dtype=bool)
    return eid, step, grasped, obj_z, carry


def test_place_steps_omits_an_episode_that_never_places():
    eid, step, grasped, obj_z, carry = _two_episode_ds()
    assert place_steps(eid, step, grasped, obj_z, carry) == {"a": 5}


def test_place_steps_uses_the_step_column_not_the_row_offset():
    eid = np.array(["a"] * 7)
    step = np.arange(100, 107)
    grasped = np.array([0, 1, 1, 1, 1, 0, 0], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.1, 1.3, 1.5, 1.5, 1.5])
    carry = np.array([0, 0, 1, 1, 1, 0, 0], dtype=bool)
    assert place_steps(eid, step, grasped, obj_z, carry) == {"a": 105}


def test_median_place_s_is_relative_to_each_episodes_own_contact():
    eid, step, grasped, obj_z, carry = _two_episode_ds()
    ds = {"episode_id": eid, "step": step, "grasped": grasped,
          "obj_pos": np.stack([np.zeros_like(obj_z), np.zeros_like(obj_z), obj_z], 1),
          "masks": {"carry": carry}}
    # only 'a' places: 5 - 1 = 4 steps = 0.2 s at 20 Hz
    assert median_place_s(ds, {"a": 1, "b": 1}) == pytest.approx(0.2)


def test_median_place_s_is_none_when_no_episode_places():
    eid = np.array(["b"] * 6)
    step = np.arange(6)
    grasped = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
    obj_z = np.array([1.0, 1.0, 1.2, 1.4, 1.5, 1.5])
    carry = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
    ds = {"episode_id": eid, "step": step, "grasped": grasped,
          "obj_pos": np.stack([np.zeros_like(obj_z), np.zeros_like(obj_z), obj_z], 1),
          "masks": {"carry": carry}}
    assert median_place_s(ds, {"b": 1}) is None


# ---------------------------------------------------------------------------
# window_available -- the camera-pixel reader's row eligibility
# ---------------------------------------------------------------------------


def test_window_available_requires_every_frame_of_the_window():
    stored = {"a": {0, 2, 4, 6, 8}}
    eid = np.array(["a", "a"])
    # step 6 needs {0, 2, 4, 6} (all stored); step 8 needs {2, 4, 6, 8} (all
    # stored); step 7 would need {1, 3, 5, 7} -- none stored
    np.testing.assert_array_equal(
        window_available(eid, np.array([6, 8]), stored, k=4, stride=2), [True, True])
    np.testing.assert_array_equal(
        window_available(eid, np.array([7, 7]), stored, k=4, stride=2), [False, False])


def test_window_available_left_clamps_early_steps():
    # step 1 with k=4 stride=2 needs max(0, 1-6..1-0) = {0, 0, 0, 1}
    stored = {"a": {0, 1}}
    np.testing.assert_array_equal(
        window_available(np.array(["a"]), np.array([1]), stored, k=4, stride=2), [True])


def test_window_available_is_false_for_an_unstored_episode():
    np.testing.assert_array_equal(
        window_available(np.array(["zz"]), np.array([0]), {"a": {0}}, k=4, stride=2),
        [False])


def test_window_available_empty_input():
    out = window_available(np.array([], dtype=object), np.array([]), {}, k=4, stride=2)
    assert out.shape == (0,)
    assert out.dtype == bool

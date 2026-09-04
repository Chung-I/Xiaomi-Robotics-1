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
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_robocasa365.mass_variation.analysis.time_curve import (
    assign_bins,
    contact_relative_steps,
    first_contact_steps,
    make_bins,
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

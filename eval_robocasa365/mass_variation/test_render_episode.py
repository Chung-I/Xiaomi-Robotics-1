# Copyright (C) 2026 Xiaomi Corporation.
"""Tests for render_episode.py's pure, sim-free pieces (fix-wave item 2).

``parse_npz_path``/``out_path_for``/``_assert_close`` need no env/sim
access; the actual replay path (``render_episode``, ``setup_condition_
physics``) is exercised live under the robocasa venv against real Phase-1
npz -- see the fix-wave report for that run's output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_robocasa365.mass_variation.render_episode import (
    ReplayDivergence,
    _assert_close,
    out_path_for,
    parse_npz_path,
)


class TestParseNpzPath:
    def test_xr1_path(self):
        path = Path(
            "output/mass_variation/phase1/PickPlaceCounterToCabinet/MassMedium/ep_37.npz"
        )
        assert parse_npz_path(path) == ("PickPlaceCounterToCabinet", "MassMedium", 37)

    def test_model_suffixed_cell_dir_strips_to_true_cell(self):
        # T7's cell_dir_for convention: "<cell>__<model>".
        path = Path(
            "output/mass_variation/phase1/"
            "PickPlaceCounterToCabinet__pi05_robocasa/MassLight/ep_7.npz"
        )
        assert parse_npz_path(path) == ("PickPlaceCounterToCabinet", "MassLight", 7)

    def test_rejects_non_ep_filename(self):
        path = Path("output/mass_variation/phase1/Cell/MassLight/episode_7.npz")
        with pytest.raises(ValueError, match="ep_<seed>.npz"):
            parse_npz_path(path)


class TestOutPathFor:
    def test_mirrors_cell_condition_seed(self):
        out_dir = Path("output/mass_variation/renders")
        npz_path = Path(
            "output/mass_variation/phase1/PickPlaceCounterToCabinet/MassHeavy/ep_39.npz"
        )
        assert out_path_for(out_dir, npz_path) == (
            out_dir / "PickPlaceCounterToCabinet" / "MassHeavy" / "ep_39.mp4"
        )


class TestAssertClose:
    def test_within_tolerance_does_not_raise(self):
        _assert_close("mass_kg", 0.6, 0.6 + 1e-10, atol=1e-9, npz_path=Path("x.npz"))

    def test_beyond_tolerance_raises_replay_divergence(self):
        with pytest.raises(ReplayDivergence, match="mass_kg"):
            _assert_close("mass_kg", 0.6, 0.7, atol=1e-9, npz_path=Path("x.npz"))

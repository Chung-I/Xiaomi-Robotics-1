# Copyright (C) 2026 Xiaomi Corporation.
"""Tests for the pure Phase-1 metrics module (Task 6, TDD).

All fixtures are synthetic npz files written into ``tmp_path`` with the same
key schema ``recorder.StepRecorder.finalize`` produces (only the keys
``metrics.py`` reads: grasped, obj_pos, liftoff_step, success, seed,
mass_kg). Every expected rate below is hand-computed in the comments.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval_robocasa365.mass_variation import metrics


def _save_ep(
    path,
    z,
    grasped,
    success,
    liftoff_step,
    seed=7,
    mass_kg=0.6,
):
    z = np.asarray(z, dtype=np.float32)
    grasped = np.asarray(grasped, dtype=bool)
    assert z.shape == grasped.shape
    obj_pos = np.zeros((z.shape[0], 3), dtype=np.float32)
    obj_pos[:, 2] = z
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        grasped=grasped,
        obj_pos=obj_pos,
        liftoff_step=np.int64(liftoff_step),
        success=np.bool_(success),
        seed=np.int64(seed),
        mass_kg=np.float64(mass_kg),
    )
    return path


def _load_ep(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class TestEpisodeMetrics:
    def test_success_episode(self, tmp_path):
        # 10 steps @20 Hz -> t_success = 0.5 s. Grasped from step 2 on,
        # z rises 0.10 m and stays up; liftoff at step 4; no release.
        z = [0.0, 0.0, 0.0, 0.02, 0.06, 0.10, 0.10, 0.10, 0.10, 0.10]
        grasped = [False, False, True, True, True, True, True, True, True, True]
        path = _save_ep(tmp_path / "ep_7.npz", z, grasped, True, 4)
        m = metrics.episode_metrics(_load_ep(path))
        assert m["success"] is True
        assert m["grasped_any"] is True
        assert m["lifted"] is True
        assert m["steps"] == 10
        assert m["t_success_s"] == pytest.approx(10 / 20.0)
        assert m["drop_after_lift"] is False
        assert m["seed"] == 7
        assert m["mass_kg"] == pytest.approx(0.6)

    def test_never_grasped_failure(self, tmp_path):
        # Never grasped, never lifted, fails: everything False, t_success NaN.
        z = [0.0] * 6
        grasped = [False] * 6
        path = _save_ep(tmp_path / "ep_8.npz", z, grasped, False, -1, seed=8)
        m = metrics.episode_metrics(_load_ep(path))
        assert m["success"] is False
        assert m["grasped_any"] is False
        assert m["lifted"] is False
        assert math.isnan(m["t_success_s"])
        assert m["drop_after_lift"] is False

    def test_drop_after_lift(self, tmp_path):
        # Liftoff at step 3; at step 6 grasped goes True->False and the
        # object falls 0.12 m over the next steps (>= 0.05 m drop threshold
        # within the window) -> drop_after_lift True.
        z = [0.0, 0.0, 0.03, 0.06, 0.10, 0.12, 0.10, 0.02, 0.00, 0.00]
        grasped = [False, True, True, True, True, True, False, False, False, False]
        path = _save_ep(tmp_path / "ep_9.npz", z, grasped, False, 3, seed=9)
        m = metrics.episode_metrics(_load_ep(path))
        assert m["lifted"] is True
        assert m["drop_after_lift"] is True
        assert m["success"] is False

    def test_gentle_place_is_not_a_drop(self, tmp_path):
        # Successful place: release at step 6 with only a 0.02 m settle
        # (< 0.05 m drop threshold) -> NOT a drop even though grasped goes
        # True->False with z falling slightly.
        z = [0.0, 0.0, 0.03, 0.06, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08]
        grasped = [False, True, True, True, True, True, False, False, False, False]
        path = _save_ep(tmp_path / "ep_10.npz", z, grasped, True, 3, seed=10)
        m = metrics.episode_metrics(_load_ep(path))
        assert m["lifted"] is True
        assert m["drop_after_lift"] is False
        assert m["success"] is True

    def test_release_before_liftoff_not_counted(self, tmp_path):
        # A failed grasp attempt (grasped True->False with z falling) BEFORE
        # liftoff never counts; only transitions strictly after liftoff_step.
        # Here liftoff never happens (liftoff_step = -1).
        z = [0.0, 0.02, 0.01, 0.0, 0.0, 0.0]
        grasped = [False, True, False, False, False, False]
        path = _save_ep(tmp_path / "ep_11.npz", z, grasped, False, -1, seed=11)
        m = metrics.episode_metrics(_load_ep(path))
        assert m["grasped_any"] is True
        assert m["lifted"] is False
        assert m["drop_after_lift"] is False

    def test_release_without_fall_not_counted(self, tmp_path):
        # Release after liftoff but the object stays supported (z flat, e.g.
        # placed on a shelf at carry height) -> not a drop.
        z = [0.0, 0.0, 0.03, 0.08, 0.10, 0.10, 0.10, 0.10]
        grasped = [False, True, True, True, True, False, False, False]
        path = _save_ep(tmp_path / "ep_12.npz", z, grasped, False, 3, seed=12)
        m = metrics.episode_metrics(_load_ep(path))
        assert m["lifted"] is True
        assert m["drop_after_lift"] is False


class TestConditionAggregation:
    def _make_condition(self, root, cell, condition, mass_kg):
        """4 episodes, hand-computed aggregates:

        seed 7:  success,   grasped, lifted, 100 steps -> t_success 5.0 s
        seed 8:  success,   grasped, lifted, 200 steps -> t_success 10.0 s
        seed 9:  fail+drop, grasped, lifted
        seed 10: fail,      never grasped, never lifted

        success_rate 2/4 = 0.5; grasp_rate 3/4 = 0.75; lift_rate 3/4 = 0.75;
        t_success mean over successes = 7.5 s; release_and_fall_rate 1/4 =
        0.25 (over ALL episodes); drop_after_lift_on_failure_rate 1/2 = 0.5
        (failures are seeds 9 and 10; only seed 9 drops).
        """
        cond_dir = root / "phase1" / cell / condition

        def flat_lift(steps):
            z = np.zeros(steps)
            z[3:] = 0.10
            grasped = np.zeros(steps, dtype=bool)
            grasped[2:] = True
            return z, grasped

        z, grasped = flat_lift(100)
        _save_ep(cond_dir / "ep_7.npz", z, grasped, True, 3, seed=7, mass_kg=mass_kg)
        z, grasped = flat_lift(200)
        _save_ep(cond_dir / "ep_8.npz", z, grasped, True, 3, seed=8, mass_kg=mass_kg)
        # fail + drop: lift to 0.10 then release at step 6, falls to 0.0
        z = np.array([0.0, 0.0, 0.03, 0.06, 0.10, 0.12, 0.10, 0.02, 0.0, 0.0])
        grasped = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        _save_ep(cond_dir / "ep_9.npz", z, grasped, False, 3, seed=9, mass_kg=mass_kg)
        # fail, never grasped
        _save_ep(
            cond_dir / "ep_10.npz",
            np.zeros(50),
            np.zeros(50, dtype=bool),
            False,
            -1,
            seed=10,
            mass_kg=mass_kg,
        )
        return cond_dir

    def test_metrics_dataframe(self, tmp_path):
        cell = "PickPlaceCounterToCabinet"
        self._make_condition(tmp_path, cell, "MassLight", 0.15)
        self._make_condition(tmp_path, cell, "MassHeavy", 1.2)

        df = metrics.metrics_dataframe(
            tmp_path / "phase1",
            cell=cell,
            conditions=["MassLight", "MassHeavy"],
            model="xr1",
        )
        assert list(df["condition"]) == ["MassLight", "MassHeavy"]
        assert set(df["model"]) == {"xr1"}
        assert set(df["cell"]) == {cell}

        row = df[df["condition"] == "MassLight"].iloc[0]
        assert row["n_episodes"] == 4
        assert row["success_rate"] == pytest.approx(0.5)
        assert row["grasp_rate"] == pytest.approx(0.75)
        assert row["lift_rate"] == pytest.approx(0.75)
        assert row["t_success_mean_s"] == pytest.approx(7.5)
        assert row["release_and_fall_rate"] == pytest.approx(0.25)
        assert row["drop_after_lift_on_failure_rate"] == pytest.approx(0.5)
        assert row["mass_kg"] == pytest.approx(0.15)

        row = df[df["condition"] == "MassHeavy"].iloc[0]
        assert row["mass_kg"] == pytest.approx(1.2)

    def test_t_success_nan_when_no_successes(self, tmp_path):
        cell = "PickPlaceCounterToCabinet"
        cond_dir = tmp_path / "phase1" / cell / "MassHeavy"
        _save_ep(
            cond_dir / "ep_7.npz",
            np.zeros(20),
            np.zeros(20, dtype=bool),
            False,
            -1,
            seed=7,
            mass_kg=1.2,
        )
        df = metrics.metrics_dataframe(
            tmp_path / "phase1", cell=cell, conditions=["MassHeavy"], model="xr1"
        )
        row = df.iloc[0]
        assert row["success_rate"] == pytest.approx(0.0)
        assert math.isnan(row["t_success_mean_s"])

    def test_drop_after_lift_on_failure_rate_nan_when_no_failures(self, tmp_path):
        # All episodes succeed (one with a release-and-fall on success, one
        # without) -> zero failures -> drop_after_lift_on_failure_rate is
        # NaN (nothing to divide by), while release_and_fall_rate still
        # counts the one release-and-fall over all episodes.
        cell = "PickPlaceCounterToCabinet"
        cond_dir = tmp_path / "phase1" / cell / "MassLight"

        def flat_lift(steps):
            z = np.zeros(steps)
            z[3:] = 0.10
            grasped = np.zeros(steps, dtype=bool)
            grasped[2:] = True
            return z, grasped

        z, grasped = flat_lift(50)
        _save_ep(cond_dir / "ep_7.npz", z, grasped, True, 3, seed=7, mass_kg=0.15)

        # Successful placement that releases into the cabinet and falls
        # 0.08 m onto the shelf below (>= the 0.05 m drop threshold) --
        # exactly the "intentional cabinet release" the reviewer flagged:
        # drop_after_lift fires, but the episode still succeeds.
        z = np.array([0.0, 0.0, 0.03, 0.06, 0.10, 0.12, 0.10, 0.02, 0.0, 0.0])
        grasped = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        _save_ep(cond_dir / "ep_8.npz", z, grasped, True, 3, seed=8, mass_kg=0.15)

        df = metrics.metrics_dataframe(
            tmp_path / "phase1", cell=cell, conditions=["MassLight"], model="xr1"
        )
        row = df.iloc[0]
        assert row["success_rate"] == pytest.approx(1.0)
        assert row["release_and_fall_rate"] == pytest.approx(0.5)
        assert math.isnan(row["drop_after_lift_on_failure_rate"])

    def test_missing_condition_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            metrics.metrics_dataframe(
                tmp_path / "phase1",
                cell="PickPlaceCounterToCabinet",
                conditions=["MassLight"],
                model="xr1",
            )

    def test_write_csv(self, tmp_path):
        import pandas as pd

        cell = "PickPlaceCounterToCabinet"
        self._make_condition(tmp_path, cell, "MassMedium", 0.6)
        df = metrics.metrics_dataframe(
            tmp_path / "phase1", cell=cell, conditions=["MassMedium"], model="xr1"
        )
        csv_path = tmp_path / "phase1" / "metrics_xr1.csv"
        metrics.write_csv(df, csv_path)
        back = pd.read_csv(csv_path)
        assert list(back["model"]) == ["xr1"]
        assert back.iloc[0]["success_rate"] == pytest.approx(0.5)

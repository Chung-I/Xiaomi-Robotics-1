# Copyright (C) 2026 Xiaomi Corporation.
"""Pure tests for the pi0.5 comparison-arm client (Plan 1, Task 7).

Parity target is the fork's OWN eval client,
``openpi-robocasa/examples/robocasa/main.py`` (@ ca4c6d7):

- state = concat(eef_pos_rel(3), eef_rot_rel QUAT(4), base_pos(3),
  base_rot QUAT(4), gripper_qpos(2)) -> 16-D, single CURRENT frame
  (main.py:133-142; no history queue anywhere in their loop).
- images: agentview_left -> "observation/image", eye_in_hand ->
  "observation/wrist_image", agentview_right -> "observation/right_image",
  each convert_to_uint8(resize_with_pad(img, 224, 224)) (main.py:118-130,
  145-151). No rotation is applied (the "rotate 180" comment at
  main.py:117 has no corresponding code).
- element keys exactly: observation/image, observation/wrist_image,
  observation/right_image, observation/state, prompt (main.py:145-151).
- actions come back already sliced to 12 dims by the server's
  RobocasaOutputs (robocasa_policy.py:127); the client re-slices
  defensively.

No websocket/sim imports in this test module; ``openpi_client`` (pure
numpy/PIL image tools) is required for resize parity.
"""

from __future__ import annotations

import numpy as np
import pytest
from openpi_client import image_tools

from eval_robocasa365.mass_variation.pi05_client import (
    PI05_ACTION_DIM,
    PI05_REPLAN_STEPS,
    PI05_RESIZE_SIZE,
    pack_pi05_element,
    slice_action_chunk,
    state_from_observation,
)


def _synthetic_observation() -> dict:
    rng = np.random.default_rng(0)
    return {
        "state.end_effector_position_relative": rng.normal(size=3).astype(np.float32),
        "state.end_effector_rotation_relative": rng.normal(size=4).astype(np.float32),
        "state.base_position": rng.normal(size=3).astype(np.float32),
        "state.base_rotation": rng.normal(size=4).astype(np.float32),
        "state.gripper_qpos": rng.normal(size=2).astype(np.float32),
        "video.robot0_agentview_left": rng.integers(0, 256, size=(256, 256, 3)).astype(np.uint8),
        "video.robot0_agentview_right": rng.integers(0, 256, size=(256, 256, 3)).astype(np.uint8),
        "video.robot0_eye_in_hand": rng.integers(0, 256, size=(256, 256, 3)).astype(np.uint8),
        "annotation.human.task_description": "pick the milk",
    }


def _obs_history_from(observation: dict, depth: int = 4) -> dict:
    """Mimic run_episode's obs_history: stacked (depth, ...) arrays whose
    LAST row is the current frame. Older rows are garbage on purpose --
    pi0.5 is single-frame and must ignore them."""
    state = state_from_observation(observation)
    states = np.stack([np.full_like(state, -99.0)] * (depth - 1) + [state])
    images = {}
    for key in (
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ):
        current = observation[key]
        garbage = np.zeros_like(current)
        images[key] = np.stack([garbage] * (depth - 1) + [current])
    return {"state": states, "images": images}


class TestStateFromObservation:
    def test_16d_order_matches_fork_client(self):
        """main.py:133-142: pos_rel, rot_rel(quat), base_pos, base_rot(quat),
        gripper -- raw quaternions, NOT the axis-angle 14-D XR1 uses."""
        obs = _synthetic_observation()
        state = state_from_observation(obs)
        assert state.shape == (16,)
        assert state.dtype == np.float32
        expected = np.concatenate(
            [
                obs["state.end_effector_position_relative"],
                obs["state.end_effector_rotation_relative"],
                obs["state.base_position"],
                obs["state.base_rotation"],
                obs["state.gripper_qpos"],
            ]
        )
        np.testing.assert_array_equal(state, expected)


class TestPackPi05Element:
    def test_keys_exactly_match_fork_element(self):
        obs = _synthetic_observation()
        history = _obs_history_from(obs)
        element = pack_pi05_element(history, "pick the milk")
        assert set(element.keys()) == {
            "observation/image",
            "observation/wrist_image",
            "observation/right_image",
            "observation/state",
            "prompt",
        }
        assert element["prompt"] == "pick the milk"

    def test_camera_mapping_and_resize_parity(self):
        """left->image, eye_in_hand->wrist, right->right_image; each is
        byte-identical to the fork's convert_to_uint8(resize_with_pad(...))."""
        obs = _synthetic_observation()
        history = _obs_history_from(obs)
        element = pack_pi05_element(history, "x")
        for element_key, obs_key in (
            ("observation/image", "video.robot0_agentview_left"),
            ("observation/wrist_image", "video.robot0_eye_in_hand"),
            ("observation/right_image", "video.robot0_agentview_right"),
        ):
            expected = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    np.ascontiguousarray(obs[obs_key]), PI05_RESIZE_SIZE, PI05_RESIZE_SIZE
                )
            )
            got = element[element_key]
            assert got.shape == (PI05_RESIZE_SIZE, PI05_RESIZE_SIZE, 3)
            assert got.dtype == np.uint8
            np.testing.assert_array_equal(got, expected)

    def test_uses_last_frame_only(self):
        """Single-frame policy: the packed state is the LAST history row
        (the current frame), garbage older rows ignored."""
        obs = _synthetic_observation()
        history = _obs_history_from(obs)
        element = pack_pi05_element(history, "x")
        np.testing.assert_array_equal(
            element["observation/state"], state_from_observation(obs)
        )
        assert not np.any(element["observation/state"] == -99.0)

    def test_accepts_single_frame_history(self):
        obs = _synthetic_observation()
        history = _obs_history_from(obs, depth=1)
        element = pack_pi05_element(history, "x")
        np.testing.assert_array_equal(
            element["observation/state"], state_from_observation(obs)
        )


class TestSliceActionChunk:
    def test_passthrough_12dim(self):
        chunk = np.arange(50 * 12, dtype=np.float64).reshape(50, 12)
        out = slice_action_chunk(chunk)
        assert out.shape == (50, PI05_ACTION_DIM)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, chunk.astype(np.float32))

    def test_slices_wider_chunks_to_12(self):
        """Defensive re-slice mirroring RobocasaOutputs' [:, :12]
        (robocasa_policy.py:127) in case a server config returns padded
        32-dim actions."""
        chunk = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
        out = slice_action_chunk(chunk)
        assert out.shape == (50, 12)
        np.testing.assert_array_equal(out, chunk[:, :12])

    def test_rejects_too_narrow(self):
        with pytest.raises(ValueError):
            slice_action_chunk(np.zeros((50, 7), dtype=np.float32))

    def test_replan_default_is_fork_default(self):
        assert PI05_REPLAN_STEPS == 5  # main.py:36


class TestRunPhase1Pi05Wiring:
    """Driver-level pure parts: gate rule + model-suffixed pathing."""

    def _df(self, light_rate: float):
        import pandas as pd

        return pd.DataFrame(
            [
                {"condition": "MassLight", "success_rate": light_rate},
                {"condition": "MassMedium", "success_rate": 0.5},
            ]
        )

    def test_pi05_gate_is_absolute_quarter(self):
        """Design addendum decision rule: pi0.5 MassLight success < 0.25
        -> STOP (no published per-task baseline)."""
        from eval_robocasa365.mass_variation.run_phase1 import evaluate_sanity_gate

        gate = evaluate_sanity_gate(self._df(0.25), "pi05_robocasa")
        assert gate["passed"] and gate["threshold"] == 0.25
        gate = evaluate_sanity_gate(self._df(0.2), "pi05_robocasa")
        assert not gate["passed"]
        assert gate["published_baseline"] is None

    def test_xr1_gate_unchanged(self):
        from eval_robocasa365.mass_variation.run_phase1 import evaluate_sanity_gate

        gate = evaluate_sanity_gate(self._df(0.66), "xr1")
        assert gate["passed"] and gate["threshold"] == pytest.approx(0.66)

    def test_cell_dir_for_model(self):
        """pi0.5 npz land in a model-suffixed cell dir; XR1's pathing (and
        its existing 105 npz) stay untouched."""
        from eval_robocasa365.mass_variation.run_phase1 import cell_dir_for

        assert cell_dir_for("PickPlaceCounterToCabinet", "xr1") == "PickPlaceCounterToCabinet"
        assert (
            cell_dir_for("PickPlaceCounterToCabinet", "pi05_robocasa")
            == "PickPlaceCounterToCabinet__pi05_robocasa"
        )

    def test_metrics_cli_cell_dir_flag(self, tmp_path):
        """metrics.py's CLI takes --cell-dir so the pi0.5 CSV is
        regenerable standalone from the model-suffixed npz dir."""
        from eval_robocasa365.mass_variation import metrics

        cond_dir = tmp_path / "Cell__pi05_robocasa" / "MassLight"
        cond_dir.mkdir(parents=True)
        np.savez(
            cond_dir / "ep_7.npz",
            grasped=np.array([False, True, True]),
            obj_pos=np.zeros((3, 3)),
            liftoff_step=1,
            success=True,
            seed=7,
            mass_kg=0.15,
        )
        csv_path = tmp_path / "metrics_pi05_robocasa.csv"
        metrics.main([
            "--phase1-root", str(tmp_path),
            "--cell", "Cell",
            "--cell-dir", "Cell__pi05_robocasa",
            "--conditions", "MassLight",
            "--model", "pi05_robocasa",
            "--csv", str(csv_path),
        ])
        import pandas as pd

        df = pd.read_csv(csv_path)
        assert list(df["cell"]) == ["Cell"]
        assert list(df["model"]) == ["pi05_robocasa"]
        assert df.iloc[0]["success_rate"] == 1.0

    def test_metrics_dataframe_cell_dir_split(self, tmp_path):
        """metrics_dataframe reads from cell_dir but labels rows with the
        true cell name."""
        from eval_robocasa365.mass_variation import metrics

        cond_dir = tmp_path / "Cell__pi05_robocasa" / "MassLight"
        cond_dir.mkdir(parents=True)
        np.savez(
            cond_dir / "ep_7.npz",
            grasped=np.array([False, True, True]),
            obj_pos=np.zeros((3, 3)),
            liftoff_step=1,
            success=True,
            seed=7,
            mass_kg=0.15,
        )
        df = metrics.metrics_dataframe(
            tmp_path,
            cell="Cell",
            conditions=["MassLight"],
            model="pi05_robocasa",
            cell_dir="Cell__pi05_robocasa",
        )
        assert list(df["cell"]) == ["Cell"]
        assert list(df["model"]) == ["pi05_robocasa"]
        assert df.iloc[0]["success_rate"] == 1.0

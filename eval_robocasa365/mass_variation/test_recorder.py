# Copyright (C) 2026 Xiaomi Corporation.
"""Tests for the ground-truth recorder (Plan 1, Task 3, Step 1).

``liftoff_step`` is pure -- exercised directly with plain arrays. The
``StepRecorder`` round trip uses a fake env stub (see ``_FakeEnv`` below)
plus an injected ``grasp_fn``, so this file needs neither robocasa nor
robosuite installed to run ``record``/``finalize`` -- only numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_robocasa365.mass_variation.recorder import StepRecorder, liftoff_step


# ---------------------------------------------------------------------------
# liftoff_step (pure)
# ---------------------------------------------------------------------------


def test_liftoff_step_no_lift_returns_minus_one():
    z = [0.5, 0.5, 0.5, 0.5]
    grasped = [True, True, True, True]
    assert liftoff_step(z, grasped, rise_m=0.05) == -1


def test_liftoff_step_rise_while_not_grasped_does_not_count():
    # rises above threshold at step 1, but ungrasped there; only becomes
    # grasped at step 3, by which point the rise (from z[0]) still clears
    # the threshold -- step 3 is the answer, not step 1.
    z = [0.50, 0.60, 0.60, 0.60]
    grasped = [False, False, False, True]
    assert liftoff_step(z, grasped, rise_m=0.05) == 3


def test_liftoff_step_rise_while_not_grasped_and_never_grasped_after():
    # same rise, but grasped never becomes True at all -> no lift-off.
    z = [0.50, 0.60, 0.60, 0.60]
    grasped = [False, False, False, False]
    assert liftoff_step(z, grasped, rise_m=0.05) == -1


def test_liftoff_step_boundary_exact_counts():
    # rise == rise_m exactly (not >) must count.
    z = [0.50, 0.55]
    grasped = [True, True]
    assert liftoff_step(z, grasped, rise_m=0.05) == 1


def test_liftoff_step_boundary_just_under_does_not_count():
    z = [0.50, 0.549999]
    grasped = [True, True]
    assert liftoff_step(z, grasped, rise_m=0.05) == -1


def test_liftoff_step_first_qualifying_step_wins():
    z = [0.50, 0.56, 0.70]
    grasped = [True, True, True]
    assert liftoff_step(z, grasped, rise_m=0.05) == 1


def test_liftoff_step_empty_returns_minus_one():
    assert liftoff_step([], [], rise_m=0.05) == -1


def test_liftoff_step_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        liftoff_step([0.5, 0.6], [True], rise_m=0.05)


# ---------------------------------------------------------------------------
# StepRecorder round trip against a fake env stub
# ---------------------------------------------------------------------------


class _FakeObj:
    def __init__(self, root_body: str) -> None:
        self.root_body = root_body


class _FakeModel:
    def __init__(self, bid: int) -> None:
        self._bid = bid
        self._joint_addr = {
            "gripper0_right_finger_joint1": 0,
            "gripper0_right_finger_joint2": 1,
        }

    def body_name2id(self, name: str) -> int:
        return self._bid

    def get_joint_qpos_addr(self, joint: str) -> int:
        return self._joint_addr[joint]


class _FakeData:
    def __init__(self, bid: int) -> None:
        n_bodies = bid + 1
        self.cfrc_ext = np.zeros((n_bodies, 6))
        self.xpos = np.zeros((n_bodies, 3))
        self.xquat = np.zeros((n_bodies, 4))
        self.xquat[:, 0] = 1.0  # identity quat (w, x, y, z) = (1, 0, 0, 0)
        self.qpos = np.array([0.0, 0.0])


class _FakeSim:
    def __init__(self, bid: int) -> None:
        self.model = _FakeModel(bid)
        self.data = _FakeData(bid)


class _FakeRobot:
    def __init__(self) -> None:
        self._eef_pos = np.zeros(3)

    @property
    def ee_force(self) -> dict:
        return {"right": np.array([0.1, 0.2, 0.3])}

    @property
    def ee_torque(self) -> dict:
        return {"right": np.array([0.01, 0.02, 0.03])}

    @property
    def _hand_pos(self) -> dict:
        return {"right": self._eef_pos}

    @property
    def _hand_orn(self) -> dict:
        return {"right": np.eye(3)}


class _FakeEnv:
    BID = 7

    def __init__(self) -> None:
        self.objects = {"obj": _FakeObj("obj_root")}
        self.sim = _FakeSim(self.BID)
        self.robots = [_FakeRobot()]

    def set_obj_z(self, z: float) -> None:
        self.sim.data.xpos[self.BID] = [0.0, 0.0, z]

    def set_eef_pos(self, pos) -> None:
        self.robots[0]._eef_pos = np.asarray(pos, dtype=float)


def test_recorder_shape_and_finalize_roundtrip(tmp_path):
    env = _FakeEnv()
    recorder = StepRecorder()

    steps = 5
    grasp_flags = [False, False, True, True, True]
    z_values = [0.50, 0.50, 0.52, 0.58, 0.60]  # rise >= 0.05 first at step 3 while grasped

    for t in range(steps):
        env.set_obj_z(z_values[t])
        env.set_eef_pos([0.1 * t, 0.0, z_values[t]])
        action = np.arange(12, dtype=np.float32) + t

        def grasp_fn(_env, _obj_name, _t=t):
            return grasp_flags[_t]

        recorder.record(env, "obj", action, grasp_fn=grasp_fn)

    assert recorder.num_steps == steps

    out_path = recorder.finalize(
        tmp_path / "ep_0.npz",
        mass_kg=0.6,
        com_offset_m=0.0,
        com_axis="y",
        seed=0,
        success=True,
    )

    data = np.load(out_path)

    # per-step array shapes, per the Task 3 brief's npz schema.
    assert data["ee_force"].shape == (steps, 3)
    assert data["ee_torque"].shape == (steps, 3)
    assert data["cfrc_obj"].shape == (steps, 6)
    assert data["obj_pos"].shape == (steps, 3)
    assert data["obj_quat"].shape == (steps, 4)
    assert data["gripper_qpos"].shape == (steps, 2)
    assert data["grasped"].shape == (steps,)
    assert data["actions"].shape == (steps, 12)
    assert data["eef_pos"].shape == (steps, 3)
    assert data["eef_rot"].shape == (steps, 3)

    # tracking-error fields (plan amendment A, the headline channel).
    assert data["commanded_delta"].shape == (steps, 6)
    assert data["achieved_eef_delta"].shape == (steps, 6)
    assert np.array_equal(data["commanded_delta"], data["actions"][:, 0:6])
    # first step has no predecessor to diff against -> zeros.
    assert np.allclose(data["achieved_eef_delta"][0], 0.0)
    # a later step's achieved delta is the diff of [eef_pos, eef_rot]
    # against the immediately preceding step.
    expected_pos_delta = np.array([0.1, 0.0, z_values[2] - z_values[1]])
    assert np.allclose(data["achieved_eef_delta"][2][:3], expected_pos_delta)

    # scalars round-trip.
    assert float(data["mass_kg"]) == pytest.approx(0.6)
    assert float(data["com_offset_m"]) == pytest.approx(0.0)
    assert str(data["com_axis"]) == "y"
    assert int(data["seed"]) == 0
    assert bool(data["success"]) is True

    # liftoff_step: grasped becomes True at index 2 (z rise from z[0]=0.50
    # is only 0.02 there); rise clears 0.05 at index 3 (0.58 - 0.50 = 0.08),
    # while still grasped -> liftoff_step == 3.
    assert int(data["liftoff_step"]) == 3
    assert int(data["liftoff_step"]) == liftoff_step(
        data["obj_pos"][:, 2], data["grasped"]
    )


def test_recorder_liftoff_minus_one_when_never_grasped(tmp_path):
    env = _FakeEnv()
    recorder = StepRecorder()

    for t in range(3):
        env.set_obj_z(0.5)
        env.set_eef_pos([0.0, 0.0, 0.5])
        recorder.record(
            env,
            "obj",
            np.zeros(12, dtype=np.float32),
            grasp_fn=lambda _env, _obj_name: False,
        )

    out_path = recorder.finalize(
        tmp_path / "ep_never_grasped.npz",
        mass_kg=1.2,
        com_offset_m=0.0,
        com_axis="y",
        seed=1,
        success=False,
    )
    data = np.load(out_path)
    assert int(data["liftoff_step"]) == -1
    assert not bool(data["grasped"].any())


def test_recorder_record_asserts_obj_name_present():
    env = _FakeEnv()
    recorder = StepRecorder()
    with pytest.raises(AssertionError):
        recorder.record(
            env, "not_obj", np.zeros(12, dtype=np.float32), grasp_fn=lambda e, n: True
        )

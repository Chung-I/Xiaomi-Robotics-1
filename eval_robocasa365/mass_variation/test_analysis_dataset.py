# Copyright (C) 2026 Xiaomi Corporation.
"""Tests for the Task 1 analysis dataset (Plan 2, mass-com-xr1).

All pure -- no sim imports, no robocasa/gym. ``TestLoadStudy`` builds tiny
synthetic Phase-1 + policy-state npz fixtures under ``tmp_path`` rather than
touching the real 210-episode corpus (that corpus is exercised by
``extract_policy_state.py``'s own bit-exactness asserts at extraction time,
run live under the robocasa venv -- see the Task 1 report).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval_robocasa365.mass_variation.analysis import dataset
from eval_robocasa365.mass_variation.pi05_client import _STATE_FIELDS, state_from_observation


# ---------------------------------------------------------------------------
# phase_masks
# ---------------------------------------------------------------------------


class TestPhaseMasks:
    def test_never_grasped_is_all_precontact(self):
        grasped = [False, False, False, False]
        obj_z = [0.5, 0.5, 0.5, 0.5]
        masks = dataset.phase_masks(grasped, liftoff_step=-1, obj_z=obj_z, init_z=0.5)
        assert list(masks["precontact"]) == [True, True, True, True]
        assert not masks["grasp"].any()
        assert not masks["carry"].any()
        assert masks["all"].all()

    def test_no_lift_episode_carry_empty(self):
        # Grasped from step 2 on, never lifted (liftoff_step == -1, the
        # recorder's convention for "never rose 5 cm").
        grasped = [False, False, True, True, True, True]
        obj_z = [0.5, 0.5, 0.5, 0.51, 0.5, 0.52]
        masks = dataset.phase_masks(grasped, liftoff_step=-1, obj_z=obj_z, init_z=0.5)
        assert list(masks["precontact"]) == [True, True, False, False, False, False]
        assert list(masks["grasp"]) == [False, False, True, True, True, True]
        assert not masks["carry"].any()  # THE fixture: no-lift -> carry empty.

    def test_drop_mid_carry_ends_carry_at_z_return_step_boundary_exact(self):
        # steps: 0,1 precontact; 2,3 grasp; liftoff at 4 (carry starts);
        # object dropped and its z crosses back below init+0.02 exactly at
        # step 7 -- carry must be True for [4,5,6] and False from 7 on
        # (boundary exact: step 7 itself, the return step, is NOT carry).
        grasped = [False, False, True, True, True, True, True, False, False]
        init_z = 0.50
        obj_z = [
            0.50,  # 0 precontact
            0.50,  # 1 precontact
            0.50,  # 2 grasp
            0.50,  # 3 grasp
            0.56,  # 4 liftoff / carry start (>= init + 0.05 rise, recorder convention)
            0.55,  # 5 carry
            0.53,  # 6 carry (still >= init + 0.02 threshold)
            0.51,  # 7 z-return step: 0.51 < init(0.50) + 0.02(0.52)? 0.51 < 0.52 True -> carry ends here
            0.50,  # 8 after return
        ]
        masks = dataset.phase_masks(grasped, liftoff_step=4, obj_z=obj_z, init_z=init_z)
        assert list(masks["precontact"]) == [True, True, False, False, False, False, False, False, False]
        assert list(masks["grasp"]) == [False, False, True, True, False, False, False, False, False]
        assert list(masks["carry"]) == [False, False, False, False, True, True, True, False, False]

    def test_carry_runs_to_episode_end_when_it_never_returns(self):
        grasped = [True, True, True, True]
        obj_z = [0.5, 0.56, 0.6, 0.65]
        masks = dataset.phase_masks(grasped, liftoff_step=1, obj_z=obj_z, init_z=0.5)
        assert list(masks["carry"]) == [False, True, True, True]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            dataset.phase_masks([True, False], liftoff_step=-1, obj_z=[0.5], init_z=0.5)


# ---------------------------------------------------------------------------
# mass_log_c
# ---------------------------------------------------------------------------


class TestMassLogC:
    def test_light_mass_exactness(self):
        result = dataset.mass_log_c(0.15)
        assert result == pytest.approx(math.log(0.25), abs=1e-12)

    def test_medium_and_heavy(self):
        assert dataset.mass_log_c(0.6) == pytest.approx(0.0, abs=1e-12)
        assert dataset.mass_log_c(1.2) == pytest.approx(math.log(2.0), abs=1e-12)

    def test_snaps_fp_noise(self):
        noisy = 0.6 + 3e-9  # measured-scalar fp noise, well under atol=1e-6.
        assert dataset.mass_log_c(noisy) == pytest.approx(0.0, abs=1e-9)

    def test_array_input(self):
        result = dataset.mass_log_c(np.array([0.15, 0.6, 1.2]))
        expected = np.array([math.log(0.25), 0.0, math.log(2.0)])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_far_from_any_level_raises(self):
        with pytest.raises(ValueError, match="not an fp-noise-only deviation"):
            dataset.mass_log_c(0.9)


# ---------------------------------------------------------------------------
# window_stack
# ---------------------------------------------------------------------------


class TestWindowStack:
    def test_shape_2d_input(self):
        x = np.arange(10 * 3).reshape(10, 3)
        out = dataset.window_stack(x, k=4, stride=2)
        assert out.shape == (10, 4, 3)

    def test_left_edge_clamp_repeats_first_frame(self):
        x = np.arange(10)
        out = dataset.window_stack(x, k=4, stride=2)
        assert list(out[0]) == [0, 0, 0, 0]

    def test_exact_indices_mid_sequence(self):
        x = np.arange(10)
        out = dataset.window_stack(x, k=4, stride=2)
        # t=5: offsets [6,4,2,0] -> idx [max(0,-1)=0, 1, 3, 5].
        assert list(out[5]) == [0, 1, 3, 5]

    def test_newest_frame_is_always_last(self):
        x = np.arange(20)
        out = dataset.window_stack(x, k=4, stride=2)
        for t in range(20):
            assert out[t][-1] == x[t]

    def test_matches_entry_sample_history_formula(self):
        # eval_robocasa365.entry.sample_history's index formula, applied to
        # a full deque of length t+1: max(0, len-1 - (length-1-index)*interval).
        x = np.arange(15)
        k, stride = 4, 2
        out = dataset.window_stack(x, k=k, stride=stride)
        for t in range(15):
            length_here = t + 1
            expected = [
                max(0, length_here - 1 - (k - 1 - i) * stride) for i in range(k)
            ]
            assert list(out[t]) == expected


# ---------------------------------------------------------------------------
# deficit_z
# ---------------------------------------------------------------------------


class TestDeficitZ:
    def test_sign_convention_commanded_minus_achieved(self):
        commanded = np.zeros((3, 6))
        achieved = np.zeros((3, 6))
        commanded[:, 2] = [-0.05, -0.02, 0.01]
        achieved[:, 2] = [-0.02, -0.02, 0.03]
        result = dataset.deficit_z(commanded, achieved)
        expected = np.array([-0.05 - (-0.02), -0.02 - (-0.02), 0.01 - 0.03], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_zero_when_achieved_matches_commanded(self):
        commanded = np.full((4, 6), 0.03)
        achieved = np.full((4, 6), 0.03)
        result = dataset.deficit_z(commanded, achieved)
        np.testing.assert_allclose(result, np.zeros(4), atol=1e-7)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            dataset.deficit_z(np.zeros((3, 6)), np.zeros((4, 6)))


# ---------------------------------------------------------------------------
# pi0.5 field-order test (brief requirement: cite lines, test against a
# synthetic obs dict) -- state_from_observation is REUSED from
# pi05_client.py (already written for the T7 pi0.5 arm and referenced by
# recorder.py's "policy_state channel" docstring), not hand-mirrored again
# here; this test pins its field order against openpi-robocasa's
# examples/robocasa/main.py:132-142 (raw quat, gripper last).
# ---------------------------------------------------------------------------


class TestPi05FieldOrder:
    def test_field_order_matches_main_py_132_142(self):
        observation = {
            "state.end_effector_position_relative": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "state.end_effector_rotation_relative": np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
            "state.base_position": np.array([8.0, 9.0, 10.0], dtype=np.float32),
            "state.base_rotation": np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
            "state.gripper_qpos": np.array([15.0, 16.0], dtype=np.float32),
        }
        state = state_from_observation(observation)
        expected = np.arange(1.0, 17.0, dtype=np.float32)  # 1..16 in field order.
        np.testing.assert_array_equal(state, expected)

    def test_field_order_tuple_matches_main_py_order(self):
        # main.py:133-142: eef_pos_rel, eef_rot_rel(quat), base_pos,
        # base_rot(quat), gripper_qpos -- pin the ORDER of the constant
        # this module concatenates over, not just the numeric result above.
        assert _STATE_FIELDS == (
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.base_position",
            "state.base_rotation",
            "state.gripper_qpos",
        )

    def test_raises_on_wrong_dimensionality(self):
        observation = {
            "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
            "state.end_effector_rotation_relative": np.zeros(4, dtype=np.float32),
            "state.base_position": np.zeros(3, dtype=np.float32),
            "state.base_rotation": np.zeros(4, dtype=np.float32),
            "state.gripper_qpos": np.zeros(1, dtype=np.float32),  # wrong: should be 2.
        }
        with pytest.raises(ValueError, match="16-D"):
            state_from_observation(observation)


# ---------------------------------------------------------------------------
# load_study (synthetic fixture, no real corpus / no sim)
# ---------------------------------------------------------------------------


def _write_phase1_episode(path, steps, mass_kg, liftoff_step, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    grasped = np.zeros(steps, dtype=bool)
    if liftoff_step >= 0:
        grasped[max(0, liftoff_step - 1):] = True
    obj_pos = np.zeros((steps, 3), dtype=np.float32)
    obj_pos[:, 2] = 0.5
    if liftoff_step >= 0:
        obj_pos[liftoff_step:, 2] = 0.6
    np.savez(
        path,
        ee_force=rng.normal(size=(steps, 3)).astype(np.float32),
        ee_torque=rng.normal(size=(steps, 3)).astype(np.float32),
        cfrc_obj=rng.normal(size=(steps, 6)).astype(np.float32),
        obj_pos=obj_pos,
        obj_quat=np.tile([0, 0, 0, 1], (steps, 1)).astype(np.float32),
        gripper_qpos=np.zeros((steps, 2), dtype=np.float32),
        grasped=grasped,
        actions=rng.normal(size=(steps, 12)).astype(np.float32),
        eef_pos=rng.normal(size=(steps, 3)).astype(np.float32),
        eef_rot=rng.normal(size=(steps, 3)).astype(np.float32),
        commanded_delta=rng.normal(size=(steps, 6)).astype(np.float32),
        achieved_eef_delta=rng.normal(size=(steps, 6)).astype(np.float32),
        liftoff_step=np.int64(liftoff_step),
        mass_kg=np.float64(mass_kg),
        com_offset_m=np.float64(0.0),
        com_axis=np.array("y"),
        seed=np.int64(seed),
        success=np.bool_(liftoff_step >= 0),
    )


def _write_policy_state_episode(path, steps, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 1000)
    np.savez(
        path,
        obs_state_14=rng.normal(size=(steps, 14)).astype(np.float32),
        obs_state_16=rng.normal(size=(steps, 16)).astype(np.float32),
    )


class TestLoadStudy:
    def _build_corpus(self, tmp_path, model="xr1", steps_by_seed=None):
        steps_by_seed = steps_by_seed or {7: 20, 8: 15}
        phase1_root = tmp_path / "phase1"
        policy_state_root = tmp_path / "policy_state"
        cell_dir = "PickPlaceCounterToCabinet" if model == "xr1" else f"PickPlaceCounterToCabinet__{model}"
        conditions = ("MassLight", "MassMedium")
        liftoffs = {"MassLight": 5, "MassMedium": -1}
        masses = {"MassLight": 0.15, "MassMedium": 0.6}
        for condition in conditions:
            for seed, steps in steps_by_seed.items():
                _write_phase1_episode(
                    phase1_root / cell_dir / condition / f"ep_{seed}.npz",
                    steps=steps, mass_kg=masses[condition],
                    liftoff_step=liftoffs[condition], seed=seed,
                )
                _write_policy_state_episode(
                    policy_state_root / model / condition / f"ep_{seed}.npz",
                    steps=steps, seed=seed,
                )
        return phase1_root, policy_state_root, conditions, list(steps_by_seed)

    def test_shapes_and_keys(self, tmp_path):
        phase1_root, policy_state_root, conditions, seeds = self._build_corpus(tmp_path)
        result = dataset.load_study(
            "xr1", phase1_root, policy_state_root, conditions=conditions, seeds=seeds,
        )
        n_total = sum(20 if s == 7 else 15 for s in seeds) * len(conditions)
        for key in (
            "episode_id", "seed", "condition", "step", "mass_kg", "mass_log_c",
            "deficit_z", "ee_force", "ee_torque", "cfrc_obj", "commanded_delta",
            "achieved_eef_delta", "grasped", "obj_pos", "policy_state_14", "policy_state_16",
        ):
            assert key in result, f"missing key {key!r}"
            assert result[key].shape[0] == n_total, f"{key} has wrong length"
        for key in ("precontact", "grasp", "carry", "all"):
            assert result["masks"][key].shape[0] == n_total
        assert result["policy_state_14"].shape[1] == 14
        assert result["policy_state_16"].shape[1] == 16
        assert result["model"] == "xr1"

    def test_pi05_model_reads_suffixed_cell_dir(self, tmp_path):
        phase1_root, policy_state_root, conditions, seeds = self._build_corpus(
            tmp_path, model="pi05_robocasa",
        )
        result = dataset.load_study(
            "pi05_robocasa", phase1_root, policy_state_root,
            conditions=conditions, seeds=seeds,
        )
        assert result["episode_id"].shape[0] > 0

    def test_mass_log_c_matches_condition(self, tmp_path):
        phase1_root, policy_state_root, conditions, seeds = self._build_corpus(tmp_path)
        result = dataset.load_study(
            "xr1", phase1_root, policy_state_root, conditions=conditions, seeds=seeds,
        )
        light_rows = result["condition"] == "MassLight"
        np.testing.assert_allclose(
            result["mass_log_c"][light_rows], math.log(0.25), atol=1e-9,
        )

    def test_no_lift_condition_has_empty_carry(self, tmp_path):
        phase1_root, policy_state_root, conditions, seeds = self._build_corpus(tmp_path)
        result = dataset.load_study(
            "xr1", phase1_root, policy_state_root, conditions=conditions, seeds=seeds,
        )
        medium_rows = result["condition"] == "MassMedium"  # liftoff_step=-1 fixture.
        assert not result["masks"]["carry"][medium_rows].any()

    def test_step_count_mismatch_raises(self, tmp_path):
        phase1_root, policy_state_root, conditions, seeds = self._build_corpus(tmp_path)
        # Corrupt one policy-state npz to have the wrong T.
        bad_path = policy_state_root / "xr1" / "MassLight" / "ep_7.npz"
        _write_policy_state_episode(bad_path, steps=999, seed=7)
        with pytest.raises(ValueError, match="diverged from the recording"):
            dataset.load_study(
                "xr1", phase1_root, policy_state_root, conditions=conditions, seeds=seeds,
            )

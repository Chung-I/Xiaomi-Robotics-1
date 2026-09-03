# Copyright (C) 2026 Xiaomi Corporation.
"""Per-step ground-truth recorder for the XR1 mass/CoM study (Plan 1, Task 3).

This module has two independently testable parts:

  - ``liftoff_step`` -- pure function, no env/sim access at all.
  - ``StepRecorder`` -- "pure-ish": ``record(env, obj_name, action, ...)`` is
    the ONLY method that touches ``env``/``sim`` (Task 3 brief requirement:
    "env access isolated in one method"); ``finalize`` is pure numpy over the
    arrays ``record`` already accumulated.

Module-import stays free of ``robocasa`` entirely -- ``record``'s only sim
package dependency (``robocasa.utils.object_utils.check_obj_grasped``) is
lazily imported inside the method itself, matching the precedent set by
``overrides.py``'s ``install_density_override``/``settle_and_gate``, and
only fires when ``grasp_fn`` is not overridden. Orientation is converted to
axis-angle with small in-module numpy helpers (``_mat_to_quat_xyzw`` /
``_quat_xyzw_to_axis_angle``), not ``robosuite.utils.transform_utils``, so
``record`` has no robosuite dependency either. The round-trip test in
``test_recorder.py`` therefore runs its fake-env stub with an injected
``grasp_fn`` and needs neither package installed -- only numpy.

Rotation delta fix (post-review, binding)
-------------------------------------------
An earlier version of ``finalize`` computed ``achieved_eef_delta``'s
rotation part by subtracting two ABSOLUTE per-step axis-angle vectors
(``eef_rot[t] - eef_rot[t-1]``). That is wrong whenever the EEF orientation
sits near theta = pi -- which it does for this study's whole episode
(``eef_rot`` norms sit at 3.02-3.14 rad throughout the Task 3 smoke
episode) -- because axis-angle has a topological discontinuity there:
``+axis*pi`` and ``-axis*pi`` encode the SAME rotation, and a real
trajectory that lingers near theta = pi can flip between the two encodings
from one step to the next with no physical motion at all. Subtracting two
such vectors cannot tell a pure encoding flip apart from an actual ~2*pi
rotation; the smoke npz had exactly this artifact (spurious ~2*pi-norm
spikes at steps 9, 109, 121, 187, 229).

The fix: ``record`` additionally keeps the RAW per-step rotation matrix
(``self._hand_orn``, not saved to the npz -- only ``eef_rot``, the absolute
axis-angle proprio copy, is), and ``finalize`` computes each step's
rotation delta from the RELATIVE rotation ``R_t @ R_{t-1}.T`` (via
``_relative_axis_angle``). A relative rotation between two consecutive 20
Hz control steps is small by construction, so it never approaches the
theta = pi singularity regardless of where the ABSOLUTE orientations sit --
well-conditioned by construction, not by luck.

Grasp detection dependency injection
-------------------------------------
``record``'s default grasp source is
``robocasa.utils.object_utils.check_obj_grasped`` (per the plan's verified
seam), lazily imported inside the method. It takes an optional ``grasp_fn``
override instead of hardcoding that import, because ``check_obj_grasped``
walks real contact geometry (``env.check_contact(gripper, obj)``) that a
lightweight fake-env test double has no reason to reimplement -- tests pass
a trivial ``lambda env, obj_name: <bool>`` there. Production callers
(``entry_mass.py``) never pass it, so they get the real seam.

cfrc_ext layout (seam correction, binding per the task brief)
----------------------------------------------------------------
MuJoCo's per-body ``cfrc_ext`` is laid out ``[torque_x, torque_y, torque_z,
force_x, force_y, force_z]`` -- torque FIRST, force LAST (verified in Task
2's ``verify_overrides.py``: index 5, not index 2, matches +mg at rest).
``cfrc_obj`` here stores that 6-vector AS-IS, unreordered: index 0:3 is
torque, index 3:6 is force, and ``cfrc_obj[..., 5]`` is Fz (+mg sign, not
-mg).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

_GRIPPER_JOINTS = ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")
_ARM = "right"


def liftoff_step(z: Any, grasped: Any, rise_m: float = 0.05) -> int:
    """First index where the object is BOTH grasped AND risen >= ``rise_m``
    from its step-0 height (the object's resting height at episode start,
    since the object does not move on the counter before it is picked up).

    A step only counts if both conditions hold AT THAT STEP:
    ``grasped[i]`` is True and ``z[i] - z[0] >= rise_m``. A rise that
    happens while ungrasped (e.g. the object gets bumped/settles) does not
    count, even if a later grasped step also clears the threshold -- the
    mask is evaluated elementwise, not via a running rise from the last
    ungrasped baseline.

    Returns -1 if the condition never holds (including on an empty input).
    """
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    grasped_arr = np.asarray(grasped, dtype=bool).reshape(-1)
    if z_arr.shape[0] == 0:
        return -1
    if z_arr.shape[0] != grasped_arr.shape[0]:
        raise ValueError(
            f"z and grasped must have the same length, got {z_arr.shape[0]} "
            f"and {grasped_arr.shape[0]}"
        )

    baseline = z_arr[0]
    rise = z_arr - baseline
    hit = grasped_arr & (rise >= rise_m)
    if not bool(hit.any()):
        return -1
    return int(np.argmax(hit))


def _mat_to_quat_xyzw(rot: Any) -> np.ndarray:
    """Rotation matrix (3, 3) -> (x, y, z, w) quaternion, trace-based
    (Shepperd w-branch, falling back to the largest-diagonal-term branch
    when w is near zero -- a near-180-degree rotation).

    Callers here pass either (a) an ABSOLUTE EEF orientation (only for
    ``eef_rot``'s per-step storage, via ``_rotmat_to_axis_angle`` below,
    matching what the policy's own proprio construction does -- see
    ``entry.observation_to_state``) or (b) a RELATIVE rotation between two
    consecutive 20 Hz control steps (``R_t @ R_{t-1}.T``, see
    ``_relative_axis_angle``), which is near-identity by construction. The
    degenerate branch below only matters for case (a); it is never
    exercised for (b).
    """
    rot = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(rot))
    w = math.sqrt(max(trace + 1.0, 0.0)) / 2.0
    if w > 1e-6:
        x = (rot[2, 1] - rot[1, 2]) / (4.0 * w)
        y = (rot[0, 2] - rot[2, 0]) / (4.0 * w)
        z = (rot[1, 0] - rot[0, 1]) / (4.0 * w)
    else:
        x = math.sqrt(max((rot[0, 0] + 1.0) / 2.0, 0.0))
        y = math.sqrt(max((rot[1, 1] + 1.0) / 2.0, 0.0))
        z = math.sqrt(max((rot[2, 2] + 1.0) / 2.0, 0.0))
        x = math.copysign(x, rot[2, 1] - rot[1, 2])
        y = math.copysign(y, rot[0, 2] - rot[2, 0])
        z = math.copysign(z, rot[1, 0] - rot[0, 1])
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_xyzw_to_axis_angle(quat: Any) -> np.ndarray:
    """(x, y, z, w) quaternion -> axis-angle exponential coordinates, via
    ``atan2(sin_half, cos_half)`` (mirrors ``entry.py``'s
    ``quat_xyzw_to_axis_angle`` -- reimplemented, not imported, so this
    module's only sim-package dependency stays the lazy
    ``check_obj_grasped`` import in ``record``; importing ``entry.py`` at
    module level would pull in its torch/transformers/imageio chain).

    ``atan2`` is well-conditioned for the ANGLE at every theta, including
    theta -> pi -- the discontinuity axis-angle has there is topological
    (``+axis*pi`` and ``-axis*pi`` are the same rotation), not a numerical
    artifact of this function, and no atan2-based fix removes it. That is
    exactly why ``finalize`` never subtracts two outputs of this function
    to get a delta -- see ``_relative_axis_angle``.
    """
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3)
    quat = quat / norm
    if quat[3] < 0:
        quat = -quat
    xyz = quat[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(sin_half, float(np.clip(quat[3], -1.0, 1.0)))
    return (xyz / sin_half) * angle


def _rotmat_to_axis_angle(rot: Any) -> np.ndarray:
    """Absolute rotation matrix -> axis-angle, for ``eef_rot`` storage."""
    return _quat_xyzw_to_axis_angle(_mat_to_quat_xyzw(rot))


def _relative_axis_angle(rot_prev: Any, rot_curr: Any) -> np.ndarray:
    """Axis-angle of the RELATIVE rotation ``rot_curr @ rot_prev.T``.

    THE fix for CRITICAL-1: a rotation DELTA must never be computed by
    subtracting two absolute axis-angle vectors where the underlying
    orientation sits near theta = pi (this study's EEF does, for the whole
    episode) -- see the module docstring's "Rotation delta fix" note.
    Composing the relative rotation directly and converting THAT to
    axis-angle sidesteps the problem entirely: the relative rotation
    between two consecutive 20 Hz steps is small by construction, so its
    own axis-angle representation is always near the origin, far from any
    singularity, independent of where ``rot_prev``/``rot_curr``
    individually sit.
    """
    rel = np.asarray(rot_curr, dtype=np.float64) @ np.asarray(rot_prev, dtype=np.float64).T
    return _quat_xyzw_to_axis_angle(_mat_to_quat_xyzw(rel))


def _gripper_qpos(env: Any) -> np.ndarray:
    """The 2-d gripper joint qpos, using the same joint names
    ``check_obj_grasped`` reads (robocasa/utils/object_utils.py:665)."""
    sim = env.sim
    return np.array(
        [
            sim.data.qpos[sim.model.get_joint_qpos_addr(joint)]
            for joint in _GRIPPER_JOINTS
        ],
        dtype=np.float32,
    )


class StepRecorder:
    """Accumulates per-step ground truth for one episode; ``finalize``
    writes it (plus the derived tracking-error and lift-off fields) to an
    npz sidecar.
    """

    def __init__(self) -> None:
        self._ee_force: list[np.ndarray] = []
        self._ee_torque: list[np.ndarray] = []
        self._cfrc_obj: list[np.ndarray] = []
        self._obj_pos: list[np.ndarray] = []
        self._obj_quat: list[np.ndarray] = []
        self._gripper_qpos: list[np.ndarray] = []
        self._grasped: list[bool] = []
        self._actions: list[np.ndarray] = []
        self._eef_pos: list[np.ndarray] = []
        self._eef_rot: list[np.ndarray] = []
        # Raw per-step rotation matrices -- NOT saved to the npz, only used
        # internally by finalize() to compute achieved_eef_delta's rotation
        # part from consecutive RELATIVE rotations (see
        # _relative_axis_angle / the module docstring's "Rotation delta
        # fix").
        self._hand_orn: list[np.ndarray] = []

    @property
    def num_steps(self) -> int:
        return len(self._actions)

    def record(self, env: Any, obj_name: str, action: Any, grasp_fn: Any = None) -> None:
        """Pull one step's ground truth off ``env`` (call this AFTER
        ``env.step`` returns, i.e. with post-step sim state) and the 12-d
        ``action`` actually applied (the RAW action, before
        ``convert_action`` -- dims 0:6 are the commanded EE delta per the
        plan amendment). ALL env/sim access for this recorder lives here.
        """
        if grasp_fn is None:
            from robocasa.utils.object_utils import check_obj_grasped as grasp_fn

        assert obj_name in env.objects, (
            f"{obj_name!r} not in env.objects (has {sorted(env.objects)}) -- "
            "the object cfg name assumption ('obj') does not hold for this env"
        )

        obj = env.objects[obj_name]
        sim = env.sim
        bid = sim.model.body_name2id(obj.root_body)
        robot = env.robots[0]

        self._ee_force.append(
            np.asarray(robot.ee_force[_ARM], dtype=np.float32).reshape(3)
        )
        self._ee_torque.append(
            np.asarray(robot.ee_torque[_ARM], dtype=np.float32).reshape(3)
        )
        self._cfrc_obj.append(
            np.asarray(sim.data.cfrc_ext[bid], dtype=np.float32).reshape(6)
        )
        self._obj_pos.append(np.asarray(sim.data.xpos[bid], dtype=np.float32).reshape(3))
        self._obj_quat.append(np.asarray(sim.data.xquat[bid], dtype=np.float32).reshape(4))
        self._gripper_qpos.append(_gripper_qpos(env))
        self._grasped.append(bool(grasp_fn(env, obj_name)))
        self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1))

        hand_pos = np.asarray(robot._hand_pos[_ARM], dtype=np.float64).reshape(3)
        hand_orn = np.asarray(robot._hand_orn[_ARM], dtype=np.float64).reshape(3, 3)
        axis_angle = _rotmat_to_axis_angle(hand_orn)

        self._eef_pos.append(hand_pos.astype(np.float32))
        self._eef_rot.append(np.asarray(axis_angle, dtype=np.float32).reshape(3))
        self._hand_orn.append(hand_orn.copy())

    def _stack(self) -> dict[str, np.ndarray]:
        if self.num_steps == 0:
            raise ValueError("StepRecorder has zero recorded steps")
        return {
            "ee_force": np.stack(self._ee_force).astype(np.float32),
            "ee_torque": np.stack(self._ee_torque).astype(np.float32),
            "cfrc_obj": np.stack(self._cfrc_obj).astype(np.float32),
            "obj_pos": np.stack(self._obj_pos).astype(np.float32),
            "obj_quat": np.stack(self._obj_quat).astype(np.float32),
            "gripper_qpos": np.stack(self._gripper_qpos).astype(np.float32),
            "grasped": np.array(self._grasped, dtype=bool),
            "actions": np.stack(self._actions).astype(np.float32),
            "eef_pos": np.stack(self._eef_pos).astype(np.float32),
            "eef_rot": np.stack(self._eef_rot).astype(np.float32),
        }

    def finalize(self, path: str | Path, **scalars: Any) -> Path:
        """Write the accumulated per-step arrays, plus the derived
        tracking-error channel (amendment A) and ``liftoff_step``, plus
        ``**scalars`` (typically ``mass_kg``, ``com_offset_m``, ``com_axis``,
        ``seed``, ``success``), to ``path`` as an npz. Returns ``path``.

        Tracking-error channel: ``commanded_delta`` is ``actions[:, 0:6]``
        (the commanded EE pos+rot delta, action dims 0:6 per
        ``robocasa.utils.env_utils.convert_action``); ``achieved_eef_delta``
        is the per-step delta against the PREVIOUS step (step 0 is zeros --
        there is no step -1 to diff against): position (dims 0:3) is a
        plain diff of ``eef_pos`` (no wraparound issue); rotation (dims 3:6)
        is ``_relative_axis_angle(R_{t-1}, R_t)`` -- NOT
        ``eef_rot[t] - eef_rot[t-1]`` (that subtracts two ABSOLUTE
        axis-angle vectors, which is wrong near theta = pi; see the module
        docstring's "Rotation delta fix").
        """
        arrays = self._stack()
        steps = arrays["actions"].shape[0]

        commanded_delta = arrays["actions"][:, 0:6].astype(np.float32)

        achieved_eef_delta = np.zeros((steps, 6), dtype=np.float32)
        if steps > 1:
            achieved_eef_delta[1:, 0:3] = arrays["eef_pos"][1:] - arrays["eef_pos"][:-1]
            for t in range(1, steps):
                achieved_eef_delta[t, 3:6] = _relative_axis_angle(
                    self._hand_orn[t - 1], self._hand_orn[t]
                )

        liftoff = liftoff_step(arrays["obj_pos"][:, 2], arrays["grasped"])

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            **arrays,
            commanded_delta=commanded_delta,
            achieved_eef_delta=achieved_eef_delta,
            liftoff_step=np.int64(liftoff),
            **scalars,
        )
        return out_path

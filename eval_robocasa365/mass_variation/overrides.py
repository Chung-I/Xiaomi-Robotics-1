# Copyright (C) 2026 Xiaomi Corporation.
"""Physics injection for the XR1 mass/CoM study (Plan 1, Task 2).

robocasa (``~/Codes/robocasa``) and robosuite are read-only for this study --
both routes below are monkeypatches from study code, no upstream edits.

Mass route (density)
---------------------
``install_density_override(density_by_objname)`` patches
``robocasa.utils.env_utils.MJCFObject`` -- **not**
``robocasa.utils.env_utils.sample_kitchen_object`` as the plan's Task 2 brief
states. That name does not exist in ``env_utils.py``: ``sample_kitchen_object``
is defined in ``robocasa.models.objects.kitchen_object_utils`` and is imported
directly into ``robocasa.environments.kitchen.kitchen`` (used by
``Kitchen.sample_object``, kitchen.py:1553) -- ``env_utils.py`` never
references that name at all (grepped; confirmed absent). What the plan's
verified-ground-truth line actually points at
(``robocasa/utils/env_utils.py:1488``) is
``MJCFObject(name=cfg["name"], **object_kwargs)`` inside ``create_obj``, and
``MJCFObject`` is a plain module-level name there
(``from robocasa.models.objects.objects import MJCFObject``). Patching that
name is the correct, minimal seam: it fires exactly once per object
construction, downstream of every RNG draw ``env.sample_object`` makes (so
overriding density never perturbs which mesh gets sampled for a given seed --
this was verified empirically in ``verify_overrides.py``), and it satisfies
the documented contract verbatim ("sets object_kwargs['density'] for the
target object name before MJCFObject construction") just via a different
attribute name than the brief's prose used.

``object_kwargs["density"]`` is not injected from nothing -- ``ObjCat`` and
``MJCFObject`` both default ``density=100`` for every category (verified in
``robocasa/models/objects/kitchen_object_utils.py`` and
``robocasa/models/objects/objects.py``), so ``mass_to_density``'s
``probe_density=100.0`` default matches what a probe (un-overridden) reset
actually used.

CoM route (body_ipos)
----------------------
``apply_com_offset`` writes ``sim.model.body_ipos`` directly, post-reset, per
object cfg name ("obj" -- preflight-verified, T1 review note: this module
asserts it explicitly in ``apply_com_offset`` rather than silently keying
into a dict that never matches). ``settle_and_gate`` then steps the env with
zero action to let contact dynamics react to the new CoM.

**Two-pass reset pattern (REQUIRED):** the settle loop inside
``Kitchen._reset_internal`` (kitchen.py:1146, ~250 substeps) runs BEFORE
``env.reset()`` ever returns control to this module, so a naive
post-reset ``body_ipos`` write only takes effect for the deltas modeled
during *this* module's own settle steps -- it never observes what the
already-completed internal settle would have done under the new CoM. The
correct sequence, on ONE live env, same seed both times (verified
deterministic: identical seed -> identical sampled mesh/scene/placement):

    1. ``env.reset(seed=s)``           # pass 1: discover mesh + authored ipos
       ``settle_and_gate(env, "obj")``  # center-CoM reference pose (offset=0)
    2. ``env.reset(seed=s)``           # pass 2: re-samples the SAME mesh
       ``apply_com_offset(env, "obj", offset_m, axis)``
       ``settle_and_gate(env, "obj", reference_pose=<pass 1's post pose>)``

Mass conditions skip all of this: density is baked in at sampling time
(*before* the internal settle loop runs), so by the time ``reset()`` returns
the settle dynamics already reflect the overridden mass/inertia -- one reset
is enough, applied via ``install_density_override`` before it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Module-level state for install/uninstall symmetry (mirrors how a single
# process runs one override "session" at a time -- matches the Task 2 usage
# pattern of install -> reset -> read -> uninstall).
_ORIGINAL_MJCF_OBJECT: Any = None

# Idempotent authored-ipos bookkeeping, keyed by mesh identity (mjcf_path) --
# NOT by body id, since hard_reset rebuilds the XML (and therefore body ids)
# every episode but the authored ipos baked into a given mesh's MJCF file is
# constant.
_AUTHORED_IPOS_CACHE: dict[str, np.ndarray] = {}


def mass_to_density(
    target_mass_kg: float, probe_mass_kg: float, probe_density: float = 100.0
) -> float:
    """Density that would give ``target_mass_kg`` for a mesh whose mass at
    ``probe_density`` (default 100, robocasa's uniform per-category default)
    is ``probe_mass_kg``.

    Pure and linear: mass scales linearly with density for fixed geometry
    (verified 10x density -> 10x mass), so
    ``target_density = target_mass_kg / probe_mass_kg * probe_density``.
    """
    if probe_mass_kg <= 0:
        raise ValueError(f"probe_mass_kg must be positive, got {probe_mass_kg!r}")
    return target_mass_kg / probe_mass_kg * probe_density


def axis_index(axis: str) -> int:
    """Map an axis label ("x"/"y"/"z") to its index into a 3-vector."""
    try:
        return _AXIS_INDEX[axis]
    except KeyError:
        raise ValueError(
            f"Unknown axis {axis!r}; expected one of {sorted(_AXIS_INDEX)}"
        ) from None


def _authored_ipos(
    cache: dict[str, np.ndarray], mesh_key: str, bid: int, body_ipos: np.ndarray
) -> np.ndarray:
    """Idempotent lookup: the first call for ``mesh_key`` snapshots (a copy
    of) ``body_ipos[bid]`` as the authored baseline; every later call for the
    same ``mesh_key`` returns that same cached baseline, unaffected by any
    writes made to ``body_ipos[bid]`` in between (e.g. a prior CoM offset).
    """
    if mesh_key not in cache:
        cache[mesh_key] = np.array(body_ipos[bid], dtype=float, copy=True)
    return np.array(cache[mesh_key], dtype=float, copy=True)


def install_density_override(density_by_objname: dict[str, float]) -> None:
    """Monkeypatch ``robocasa.utils.env_utils.MJCFObject`` so that objects
    whose cfg ``name`` is a key of ``density_by_objname`` are constructed
    with ``density=density_by_objname[name]`` instead of whatever density
    ``env.sample_object`` returned. See the module docstring for why this
    (and not ``sample_kitchen_object``) is the correct seam.

    Idempotency: raises if an override is already installed -- call
    ``uninstall_density_override()`` first (mirrors the two-pass reset
    pattern: one override "session" per verification/episode).
    """
    global _ORIGINAL_MJCF_OBJECT

    import robocasa.utils.env_utils as env_utils

    if _ORIGINAL_MJCF_OBJECT is not None:
        raise RuntimeError(
            "density override already installed; call "
            "uninstall_density_override() first"
        )

    original_mjcf_object = env_utils.MJCFObject

    def _patched_mjcf_object(name: str, **kwargs: Any):
        if name in density_by_objname:
            kwargs = dict(kwargs)
            kwargs["density"] = float(density_by_objname[name])
        return original_mjcf_object(name=name, **kwargs)

    _ORIGINAL_MJCF_OBJECT = original_mjcf_object
    env_utils.MJCFObject = _patched_mjcf_object


def uninstall_density_override() -> None:
    """Restore ``robocasa.utils.env_utils.MJCFObject``. No-op if nothing is
    installed."""
    global _ORIGINAL_MJCF_OBJECT

    if _ORIGINAL_MJCF_OBJECT is None:
        return

    import robocasa.utils.env_utils as env_utils

    env_utils.MJCFObject = _ORIGINAL_MJCF_OBJECT
    _ORIGINAL_MJCF_OBJECT = None


def apply_com_offset(env: Any, obj_name: str, offset_m: float, axis: str) -> dict:
    """Write an absolute CoM offset (from the mesh's authored ``body_ipos``)
    onto ``obj_name``'s body, post-reset.

    Resolves ``bid`` fresh every call (required: ``hard_reset=True`` rebuilds
    the XML, and therefore body ids, every episode). Asserts ``obj_name`` is
    actually a live object on ``env`` -- the T1 review note: the object cfg
    name ("obj") is an assumption, not a guarantee, and a silent no-op keyed
    into a dict that never matches would be worse than a loud failure here.

    Idempotent: the write is always computed as
    ``authored_ipos[axis] + offset_m`` (not ``current_ipos[axis] + offset_m``)
    against a cached authored baseline (see ``_authored_ipos``), so calling
    this twice for the same mesh with the same offset does not double-apply.
    """
    assert obj_name in env.objects, (
        f"{obj_name!r} not in env.objects (has {sorted(env.objects)}) -- "
        "the object cfg name assumption ('obj') does not hold for this env"
    )

    axis_idx = axis_index(axis)
    obj = env.objects[obj_name]
    model = env.sim.model
    bid = model.body_name2id(obj.root_body)
    mesh_key = str(obj.mjcf_path)

    authored = _authored_ipos(_AUTHORED_IPOS_CACHE, mesh_key, bid, model.body_ipos)
    new_ipos = authored.copy()
    new_ipos[axis_idx] = authored[axis_idx] + offset_m
    model.body_ipos[bid] = new_ipos
    env.sim.forward()

    mass = float(model.body_mass[bid])
    return {"bid": int(bid), "mass": mass, "ipos": new_ipos.tolist()}


def _quat_angle_deg(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    """Angle (degrees) between two orientations given as wxyz quaternions."""
    a = np.asarray(quat_a, dtype=float)
    b = np.asarray(quat_b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def settle_and_gate(
    env: Any,
    obj_name: str,
    n_steps: int = 20,
    pos_tol_m: float = 1e-3,
    rot_tol_deg: float = 0.5,
    reference_pose: dict | None = None,
) -> dict:
    """Step ``env`` ``n_steps`` with a zero action, then gate the resulting
    object pose.

    ``reference_pose`` (``{"pos": (3,), "quat": (4,)}``), when given, is
    typically the post-settle pose from a CENTER-CoM (offset=0) reset at the
    same seed -- the "post-settle pose delta vs center" gate from the design
    doc (Phase-0 precedent: CoM offset must be settle-invisible). When
    omitted, the reference is this call's own pre-settle pose (captured right
    before stepping, i.e. immediately after ``apply_com_offset`` +
    ``sim.forward()``) -- a self-check for whether the offset alone (without
    even a comparison run) visibly perturbs the object once real dynamics
    (not just ``forward()``) execute.

    ``passed`` is True iff both the position delta (meters) and the
    orientation delta (degrees) are strictly under their tolerances.
    """
    from robocasa.utils.env_utils import convert_action

    assert obj_name in env.objects, (
        f"{obj_name!r} not in env.objects (has {sorted(env.objects)})"
    )

    obj = env.objects[obj_name]
    bid = env.sim.model.body_name2id(obj.root_body)

    pre_pos = np.array(env.sim.data.xpos[bid], dtype=float, copy=True)
    pre_quat = np.array(env.sim.data.xquat[bid], dtype=float, copy=True)

    zero_action = convert_action(np.zeros(12, dtype=np.float32))
    for _ in range(n_steps):
        env.step(zero_action)

    post_pos = np.array(env.sim.data.xpos[bid], dtype=float, copy=True)
    post_quat = np.array(env.sim.data.xquat[bid], dtype=float, copy=True)

    if reference_pose is not None:
        ref_pos = np.asarray(reference_pose["pos"], dtype=float)
        ref_quat = np.asarray(reference_pose["quat"], dtype=float)
    else:
        ref_pos, ref_quat = pre_pos, pre_quat

    pos_delta_m = float(np.linalg.norm(post_pos - ref_pos))
    rot_delta_deg = _quat_angle_deg(post_quat, ref_quat)
    passed = bool(pos_delta_m < pos_tol_m and rot_delta_deg < rot_tol_deg)

    return {
        "bid": int(bid),
        "pre_pos": pre_pos.tolist(),
        "pre_quat": pre_quat.tolist(),
        "post_pos": post_pos.tolist(),
        "post_quat": post_quat.tolist(),
        "pos_delta_m": pos_delta_m,
        "rot_delta_deg": rot_delta_deg,
        "pos_tol_m": pos_tol_m,
        "rot_tol_deg": rot_tol_deg,
        "passed": passed,
    }

# Copyright (C) 2026 Xiaomi Corporation.
"""Task 2, Step 3: verification script for the physics-injection module.

Sim-touching (imports robocasa/gymnasium; kept out of overrides.py's pure
parts). Exercises both injection routes on the primary cell
(PickPlaceCounterToCabinet x milk, per Task 1's preflight) at a single
matched seed and checks:

  (a) density route: request 0.60 kg -> measured body_mass within 1%; same
      mesh + same body_ipos as the (un-overridden) 100-density baseline at
      the same seed.
  (b) CoM route: offset +/-0.02 m -> body_ipos moved exactly (from the
      authored baseline), mass unchanged, data.xipos propagates; settle
      gate (pose delta vs a center-CoM [offset=0] reference at the same
      seed) passes on "y", or the script falls back to "x" and records
      whichever axis it actually used.
  (c) matched-pair check: the density route and the CoM route, run at the
      SAME seed, sample the identical mesh (name equality asserted) --
      cross-checks that installing/uninstalling the density override does
      not perturb env.rng's draw for a fresh env.
  (d) cfrc_ext[bid] z-component ~= -mg within 5% after settle, at the
      overridden 0.60 kg mass.

Run with:
    MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python \
        eval_robocasa365/mass_variation/verify_overrides.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_robocasa365.mass_variation.overrides import (
    apply_com_offset,
    axis_index,
    install_density_override,
    mass_to_density,
    settle_and_gate,
    uninstall_density_override,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "output" / "mass_variation" / "verify_overrides.json"

PRIMARY_ENV = "PickPlaceCounterToCabinet"
PRIMARY_CATEGORY = "milk"
SEED = 0

TARGET_MASS_KG = 0.60
MASS_TOL_PCT = 1.0
COM_OFFSET_M = 0.02
POS_TOL_M = 1e-3
ROT_TOL_DEG = 0.5
SETTLE_STEPS = 20
CFRC_TOL_PCT = 5.0


def _obj_state(env: Any) -> dict:
    """Snapshot of the "obj" object's identity/physics for the current
    reset. Asserts "obj" exists in env.objects (T1 review note: the object
    cfg name is an assumption, made explicit here rather than silently
    keyed into a dict that never matches)."""
    assert "obj" in env.objects, (
        f"'obj' not in env.objects (has {sorted(env.objects)}) -- the T1 "
        "object-cfg-name assumption does not hold for this env"
    )
    obj = env.objects["obj"]
    bid = env.sim.model.body_name2id(obj.root_body)
    return {
        "bid": int(bid),
        "mesh": str(obj.mjcf_path),
        "mass_kg": float(env.sim.model.body_mass[bid]),
        "ipos": np.array(env.sim.model.body_ipos[bid], dtype=float, copy=True),
        "xipos": np.array(env.sim.data.xipos[bid], dtype=float, copy=True),
    }


def verify_density_route(gym: Any) -> dict:
    result: dict[str, Any] = {}
    env = gym.make(
        f"robocasa/{PRIMARY_ENV}",
        split="pretrain",
        obj_groups=PRIMARY_CATEGORY,
        seed=SEED,
    )
    try:
        # pass 1: baseline, no override installed.
        env.reset(seed=SEED)
        baseline = _obj_state(env)
        result["baseline"] = {
            "mesh": baseline["mesh"],
            "mass_kg": baseline["mass_kg"],
            "ipos": baseline["ipos"].tolist(),
        }

        target_density = mass_to_density(
            TARGET_MASS_KG, baseline["mass_kg"], probe_density=100.0
        )
        result["target_density"] = target_density

        install_density_override({"obj": target_density})
        try:
            # pass 2: same seed, override active.
            env.reset(seed=SEED)
            overridden = _obj_state(env)

            mass_err_pct = (
                abs(overridden["mass_kg"] - TARGET_MASS_KG) / TARGET_MASS_KG * 100.0
            )
            mesh_match = overridden["mesh"] == baseline["mesh"]
            ipos_match = bool(
                np.allclose(overridden["ipos"], baseline["ipos"], atol=1e-6)
            )

            result["overridden"] = {
                "mesh": overridden["mesh"],
                "mass_kg": overridden["mass_kg"],
                "ipos": overridden["ipos"].tolist(),
                "mass_err_pct": mass_err_pct,
                "mesh_match": mesh_match,
                "ipos_match": ipos_match,
            }

            assert mass_err_pct < MASS_TOL_PCT, (
                f"density route: mass error {mass_err_pct:.3f}% >= "
                f"{MASS_TOL_PCT}% tolerance"
            )
            assert mesh_match, (
                f"density route: mesh changed across the same seed "
                f"({baseline['mesh']!r} -> {overridden['mesh']!r})"
            )
            assert ipos_match, (
                "density route: body_ipos differs from the 100-density "
                f"baseline (baseline={baseline['ipos']}, "
                f"overridden={overridden['ipos']})"
            )

            # (d) cfrc_ext Fz ~= mg within 5%, after settle, at 0.60 kg.
            #
            # Correction to the brief: MuJoCo's cfrc_ext layout per body is
            # [torque_x, torque_y, torque_z, force_x, force_y, force_z] --
            # torque FIRST, force last (a well-known MuJoCo gotcha) -- so Fz
            # is index 5, not index 2 (index 2 is torque_z). Verified
            # empirically (debug script, milk_0 default density): index 5 at
            # rest reads 0.534957, matching default mass 0.05453 kg * 9.81 =
            # 0.53496 almost exactly, and with a POSITIVE sign -- so the
            # correct expectation is Fz ~= +mg, not the brief's "-mg". This
            # matches the plan's own ground-truth line ("object wrench
            # cfrc_ext[bid] (verified = mg)", no minus sign) over the Task 2
            # brief's "-mg" wording.
            settle = settle_and_gate(
                env,
                "obj",
                n_steps=SETTLE_STEPS,
                pos_tol_m=POS_TOL_M,
                rot_tol_deg=ROT_TOL_DEG,
            )
            bid = overridden["bid"]
            fz = float(env.sim.data.cfrc_ext[bid][5])
            gravity = float(abs(env.sim.model.opt.gravity[2]))
            mg = overridden["mass_kg"] * gravity
            expected_fz = mg
            cfrc_err_pct = abs(fz - expected_fz) / mg * 100.0

            result["cfrc_check"] = {
                "fz": fz,
                "gravity": gravity,
                "mg": mg,
                "expected_fz": expected_fz,
                "cfrc_err_pct": cfrc_err_pct,
                "settle": settle,
            }

            assert cfrc_err_pct < CFRC_TOL_PCT, (
                f"density route: cfrc_ext Fz={fz:.4f} vs expected "
                f"{expected_fz:.4f} (mg={mg:.4f}), error "
                f"{cfrc_err_pct:.2f}% >= {CFRC_TOL_PCT}% tolerance"
            )
        finally:
            uninstall_density_override()
    finally:
        env.close()

    result["passed"] = True
    return result


def verify_com_route(gym: Any) -> dict:
    result: dict[str, Any] = {}
    env = gym.make(
        f"robocasa/{PRIMARY_ENV}",
        split="pretrain",
        obj_groups=PRIMARY_CATEGORY,
        seed=SEED,
    )
    try:
        # pass 1: discover mesh + authored physics, AND the center-CoM
        # (offset=0) settle reference this gate compares against.
        env.reset(seed=SEED)
        center_state = _obj_state(env)
        center_settle = settle_and_gate(
            env,
            "obj",
            n_steps=SETTLE_STEPS,
            pos_tol_m=POS_TOL_M,
            rot_tol_deg=ROT_TOL_DEG,
        )
        reference_pose = {
            "pos": center_settle["post_pos"],
            "quat": center_settle["post_quat"],
        }
        result["mesh"] = center_state["mesh"]
        result["authored_ipos"] = center_state["ipos"].tolist()
        result["center_settle"] = center_settle

        axis_attempts = []
        chosen_axis = None
        chosen_offset_result = None
        chosen_gate = None

        for axis in ("y", "x"):
            # pass 2 (redone per axis attempt): same seed -> same mesh.
            env.reset(seed=SEED)
            pre_state = _obj_state(env)

            offset_result = apply_com_offset(env, "obj", COM_OFFSET_M, axis)
            axis_idx = axis_index(axis)

            ipos_moved_exactly = bool(
                np.isclose(
                    offset_result["ipos"][axis_idx],
                    pre_state["ipos"][axis_idx] + COM_OFFSET_M,
                    atol=1e-9,
                )
            )
            mass_unchanged = bool(
                np.isclose(offset_result["mass"], pre_state["mass_kg"], atol=1e-12)
            )
            post_offset_xipos = np.array(
                env.sim.data.xipos[offset_result["bid"]], dtype=float, copy=True
            )
            xipos_delta = float(
                np.linalg.norm(post_offset_xipos - pre_state["xipos"])
            )
            # xipos must have moved by roughly the offset magnitude (not
            # exact equality -- body_ipos is in the body frame, xipos is in
            # world frame, related by the body's orientation, close to
            # identity for these upright objects but not asserted exact).
            xipos_propagated = bool(0.5 * COM_OFFSET_M < xipos_delta < 1.5 * COM_OFFSET_M)

            gate = settle_and_gate(
                env,
                "obj",
                n_steps=SETTLE_STEPS,
                pos_tol_m=POS_TOL_M,
                rot_tol_deg=ROT_TOL_DEG,
                reference_pose=reference_pose,
            )

            attempt = {
                "axis": axis,
                "offset_result": offset_result,
                "ipos_moved_exactly": ipos_moved_exactly,
                "mass_unchanged": mass_unchanged,
                "xipos_delta_m": xipos_delta,
                "xipos_propagated": xipos_propagated,
                "gate": gate,
            }
            axis_attempts.append(attempt)

            assert ipos_moved_exactly, (
                f"CoM route ({axis}): body_ipos did not move exactly by "
                f"{COM_OFFSET_M} from the authored baseline"
            )
            assert mass_unchanged, (
                f"CoM route ({axis}): mass changed "
                f"({pre_state['mass_kg']} -> {offset_result['mass']})"
            )
            assert xipos_propagated, (
                f"CoM route ({axis}): data.xipos delta {xipos_delta:.5f} m "
                f"is not within [0.5, 1.5] x {COM_OFFSET_M} m of the offset "
                "-- forward() did not propagate the ipos write"
            )

            if gate["passed"]:
                chosen_axis = axis
                chosen_offset_result = offset_result
                chosen_gate = gate
                break

        result["axis_attempts"] = axis_attempts
        result["chosen_axis"] = chosen_axis
        result["gate_passed"] = chosen_axis is not None

        if chosen_axis is None:
            # Neither axis passed the settle gate -- record failure as data,
            # per the plan's convention that a real negative result is not
            # a bug. Fall back to "y" (the design doc's stated default) for
            # downstream bookkeeping, but gate_passed stays False.
            chosen_axis = "y"
            chosen_offset_result = axis_attempts[0]["offset_result"]
            chosen_gate = axis_attempts[0]["gate"]

        result["offset_result"] = chosen_offset_result
        result["settle_gate"] = chosen_gate
    finally:
        env.close()

    result["passed"] = result["gate_passed"]
    return result


def main() -> None:
    import gymnasium as gym
    import robocasa  # noqa: F401

    density_result = verify_density_route(gym)
    com_result = verify_com_route(gym)

    # (c) matched-pair check: same seed, both routes sample the identical
    # mesh.
    matched_pair_ok = bool(density_result["baseline"]["mesh"] == com_result["mesh"])
    assert matched_pair_ok, (
        "matched-pair check: density route sampled "
        f"{density_result['baseline']['mesh']!r} but CoM route sampled "
        f"{com_result['mesh']!r} at the same seed {SEED}"
    )

    overall_passed = bool(
        density_result["passed"] and com_result["passed"] and matched_pair_ok
    )

    report = {
        "primary_env": PRIMARY_ENV,
        "primary_category": PRIMARY_CATEGORY,
        "seed": SEED,
        "density_route": density_result,
        "com_route": com_result,
        "matched_pair": {
            "density_route_mesh": density_result["baseline"]["mesh"],
            "com_route_mesh": com_result["mesh"],
            "matched": matched_pair_ok,
        },
        "chosen_com_axis": com_result["chosen_axis"],
        "overall_passed": overall_passed,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Wrote {OUTPUT_PATH}\n")
    print("=== Task 2 verify_overrides summary ===")
    print(f"{'check':<30}{'result':<40}")
    print(
        f"{'(a) density mass err %':<30}"
        f"{density_result['overridden']['mass_err_pct']:.3f}"
    )
    print(f"{'(a) mesh match':<30}{density_result['overridden']['mesh_match']}")
    print(f"{'(a) ipos match':<30}{density_result['overridden']['ipos_match']}")
    print(
        f"{'(d) cfrc_ext err %':<30}"
        f"{density_result['cfrc_check']['cfrc_err_pct']:.3f}"
    )
    print(f"{'(b) chosen CoM axis':<30}{com_result['chosen_axis']}")
    print(f"{'(b) settle gate passed':<30}{com_result['gate_passed']}")
    print(f"{'(c) matched-pair mesh':<30}{matched_pair_ok}")
    print(f"{'OVERALL':<30}{overall_passed}")


if __name__ == "__main__":
    main()

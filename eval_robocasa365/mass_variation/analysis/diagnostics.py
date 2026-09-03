# Copyright (C) 2026 Xiaomi Corporation.
"""Plan-2 Task 2 (review IMPORTANT-1): regenerate the load-bearing
diagnosis numbers behind amendment B and the K_eff frame verdict.

These numbers first appeared only in the Task-2 report prose; this script
makes them reproducible from the corpora (Task 6 cites them from here /
from ``diagnostics.json``). Per model it computes:

1. **Per-condition carry channel means** (mask: ``carry``): wrist |F|,
   |tau|, EE-frame F_z, base-frame force vector, world-frame force vector
   (rotations via ``certificates.derived_force_features`` -- single
   source), and ``cfrc_obj`` vertical force vs the expected ``m*g``
   (g = 9.81 m/s^2, the MuJoCo default). The |F| mass-monotonicity and
   the cfrc ~= mg agreement are amendment B's trigger facts.
2. **Axis-correlation matrix** (mask: ``all`` steps) between
   ``commanded_delta[:, 0:3]`` and ``achieved_eef_delta[:, 0:3]`` --
   |corr| per (commanded axis, achieved axis) pair, plus the base-yaw
   spread (std of ``policy_state_14[:, 13]``, the z component of the base
   axis-angle). Diagonal dominance despite a ~1.5 rad yaw spread is the
   frame arbiter: recorder ``eef_pos`` is BASE-frame (same frame as the
   commanded OSC action), not world-frame as its docstring claimed.
3. **Units/std mismatch** (masks: ``carry`` and ``all``): std of
   ``commanded_delta_z`` (normalized action units) vs
   ``achieved_eef_delta_z`` (meters), and the effective tracking gain from
   the OLS ``achieved_z ~ commanded_z`` (mask: ``all``) -- documents why
   ``deficit_z`` is dominated by the commanded term (certificates
   ``K_EFF_UNITS_NOTE``).

Run (robocasa venv):
    ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.analysis.diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eval_robocasa365.mass_variation.analysis.certificates import (
    DEFAULT_PHASE1_ROOT,
    DEFAULT_POLICY_STATE_ROOT,
    MODELS,
    derived_force_features,
    sanitize_json,
)
from eval_robocasa365.mass_variation.analysis.dataset import (
    MASS_CONDITIONS,
    load_study,
)

G_M_S2 = 9.81  # MuJoCo default gravity magnitude
DEFAULT_OUT = "output/mass_variation/analysis/diagnostics.json"

# recorder.py cfrc_obj layout: [torque (0:3), force (3:6)] -> vertical force
CFRC_FZ_COL = 5


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """slope, intercept, R^2 of y ~ x."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = slope * x + intercept
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def model_diagnostics(d: dict) -> dict:
    """All three diagnostic blocks for one loaded corpus (see module
    docstring). Pure given ``load_study``'s dict."""
    carry = d["masks"]["carry"]
    derived = derived_force_features(d["ee_force"], d["ee_torque"],
                                     d["policy_state_14"])
    f_norm, tau_norm = derived[:, 0], derived[:, 1]
    f_world, f_base = derived[:, 2:5], derived[:, 5:8]

    per_condition = {}
    for c in MASS_CONDITIONS:
        m = carry & (d["condition"] == c)
        mass = float(d["mass_kg"][m].mean())
        per_condition[c] = {
            "mask": "carry",
            "n": int(m.sum()),
            "mass_kg_mean": mass,
            "abs_force_mean_N": float(f_norm[m].mean()),
            "abs_torque_mean_Nm": float(tau_norm[m].mean()),
            "ee_frame_fz_mean_N": float(d["ee_force"][m, 2].mean()),
            "base_frame_force_mean_N": [float(v) for v in f_base[m].mean(0)],
            "world_frame_force_mean_N": [float(v) for v in f_world[m].mean(0)],
            "cfrc_obj_fz_mean_N": float(d["cfrc_obj"][m, CFRC_FZ_COL].mean()),
            "expected_mg_N": mass * G_M_S2,
        }

    cd = np.asarray(d["commanded_delta"], dtype=np.float64)
    ad = np.asarray(d["achieved_eef_delta"], dtype=np.float64)
    axis_corr = [[float(abs(np.corrcoef(cd[:, j], ad[:, i])[0, 1]))
                  for j in range(3)] for i in range(3)]  # rows: achieved axis

    gain, gain_b, gain_r2 = _ols(ad[:, 2], cd[:, 2])
    return {
        "per_condition_carry": per_condition,
        "axis_correlation": {
            "mask": "all",
            "abs_corr_achieved_axis_by_commanded_axis": axis_corr,
            "base_yaw_std_rad": float(np.std(d["policy_state_14"][:, 13])),
            "note": ("diagonal dominance despite the cross-episode base-yaw "
                     "spread => achieved_eef_delta (from recorder eef_pos "
                     "diffs) shares the commanded action's BASE frame; "
                     "world-frame deltas would mix x/y across episode yaws"),
        },
        "units": {
            "commanded_delta_z_std_all": float(cd[:, 2].std()),
            "commanded_delta_z_std_carry": float(cd[carry, 2].std()),
            "achieved_eef_delta_z_std_m_all": float(ad[:, 2].std()),
            "achieved_eef_delta_z_std_m_carry": float(ad[carry, 2].std()),
            "tracking_gain_m_per_action_unit": gain,
            "tracking_gain_intercept_m": gain_b,
            "tracking_gain_r2": gain_r2,
            "mask": "all (gain fit); stds reported for all and carry",
            "note": ("commanded_delta is the raw normalized OSC action "
                     "(dimensionless, ~+-1); achieved_eef_delta is metric -- "
                     "deficit_z therefore is dominated by the commanded term"),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1-root", default=DEFAULT_PHASE1_ROOT)
    ap.add_argument("--policy-state-root", default=DEFAULT_POLICY_STATE_ROOT)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    result = {"g_m_s2": G_M_S2}
    for model in args.models:
        d = load_study(model, args.phase1_root, args.policy_state_root)
        diag = model_diagnostics(d)
        result[model] = diag
        print(f"===== {model} =====", flush=True)
        for c, v in diag["per_condition_carry"].items():
            print(f"  [carry] {c}: |F|={v['abs_force_mean_N']:.2f} N  "
                  f"|tau|={v['abs_torque_mean_Nm']:.3f} Nm  "
                  f"ee Fz={v['ee_frame_fz_mean_N']:+.3f} N  "
                  f"F_base={np.round(v['base_frame_force_mean_N'], 3)}  "
                  f"F_world={np.round(v['world_frame_force_mean_N'], 3)}  "
                  f"cfrc_fz={v['cfrc_obj_fz_mean_N']:.2f} vs "
                  f"mg={v['expected_mg_N']:.2f} N (n={v['n']})", flush=True)
        ac = diag["axis_correlation"]
        print(f"  [all] |corr| achieved x commanded (rows=achieved axis):",
              flush=True)
        for i, row in enumerate(ac["abs_corr_achieved_axis_by_commanded_axis"]):
            print(f"        {'xyz'[i]}: {['%.3f' % v for v in row]}", flush=True)
        print(f"  [all] base yaw std = {ac['base_yaw_std_rad']:.3f} rad", flush=True)
        u = diag["units"]
        print(f"  [units] cmd_z std carry={u['commanded_delta_z_std_carry']:.4f} "
              f"(action units) vs ach_z std carry="
              f"{u['achieved_eef_delta_z_std_m_carry']:.5f} m; tracking gain "
              f"{u['tracking_gain_m_per_action_unit']:.5f} m/unit "
              f"(R2={u['tracking_gain_r2']:.3f})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(sanitize_json(result), indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

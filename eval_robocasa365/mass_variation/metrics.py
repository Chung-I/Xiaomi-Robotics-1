# Copyright (C) 2026 Xiaomi Corporation.
"""Pure Phase-1 metrics for the XR1 mass/CoM study (Plan 1, Task 6).

Reads the per-episode npz sidecars ``recorder.StepRecorder.finalize`` wrote
(``output/mass_variation/phase1/<cell>/<condition>/ep_<seed>.npz``) and
reduces them to a tidy per-condition DataFrame -> CSV. No sim imports;
numpy + pandas only.

Per-episode definitions (control rate 20 Hz -> 1 step = 0.05 s):

- ``success``       -- the npz's own ``success`` scalar (env ``info``).
- ``grasped_any``   -- any step with ``grasped`` True.
- ``lifted``        -- ``liftoff_step >= 0`` (recorder's grasped-AND-risen
                       >= 0.05 m criterion).
- ``t_success_s``   -- episode steps / 20 Hz if success, else NaN (a
                       successful episode terminates at the success step,
                       so steps IS the time-to-success).
- ``drop_after_lift`` -- after liftoff, a grasped True->False transition at
  step t whose object then FALLS: min z over ``[t, t+drop_window)`` is at
  least ``drop_m`` below z[t-1] (the last grasped height). The 0.05 m
  default threshold is symmetric with the recorder's liftoff rise
  threshold and exists to keep a deliberate place (a ~1-2 cm settle onto
  the shelf after release) from counting as a drop; a real drop from carry
  height falls well past 5 cm.

Aggregation (one row per condition): every ``*_rate`` uses ALL episodes of
the condition as denominator; ``t_success_mean_s`` averages over successful
episodes only (NaN when there are none).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

CONTROL_HZ = 20.0
DROP_M = 0.05
DROP_WINDOW_STEPS = 20  # 1 s @ 20 Hz


def episode_metrics(
    ep: Mapping[str, Any],
    drop_m: float = DROP_M,
    drop_window: int = DROP_WINDOW_STEPS,
    control_hz: float = CONTROL_HZ,
) -> dict[str, Any]:
    """Per-episode metrics from one npz's (already loaded) arrays/scalars.

    ``ep`` needs keys: grasped (T,), obj_pos (T, 3), liftoff_step, success,
    seed, mass_kg.
    """
    grasped = np.asarray(ep["grasped"], dtype=bool).reshape(-1)
    z = np.asarray(ep["obj_pos"], dtype=float).reshape(-1, 3)[:, 2]
    if grasped.shape[0] != z.shape[0]:
        raise ValueError(
            f"grasped and obj_pos length mismatch: {grasped.shape[0]} vs {z.shape[0]}"
        )
    steps = int(grasped.shape[0])
    success = bool(np.asarray(ep["success"]))
    liftoff = int(np.asarray(ep["liftoff_step"]))
    lifted = liftoff >= 0

    drop_after_lift = False
    if lifted:
        # Release transitions strictly AFTER liftoff: grasped[t-1] True,
        # grasped[t] False, for t > liftoff.
        for t in range(max(liftoff + 1, 1), steps):
            if grasped[t - 1] and not grasped[t]:
                z_min = float(np.min(z[t : min(t + drop_window, steps)]))
                if z[t - 1] - z_min >= drop_m:
                    drop_after_lift = True
                    break

    return {
        "seed": int(np.asarray(ep["seed"])),
        "steps": steps,
        "success": success,
        "grasped_any": bool(grasped.any()),
        "lifted": lifted,
        "liftoff_step": liftoff,
        "t_success_s": steps / control_hz if success else float("nan"),
        "drop_after_lift": drop_after_lift,
        "mass_kg": float(np.asarray(ep["mass_kg"])),
    }


def load_episode(npz_path: str | Path) -> dict[str, Any]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def condition_episode_frame(cond_dir: str | Path) -> pd.DataFrame:
    """One row per ``ep_*.npz`` in ``cond_dir`` (sorted by seed)."""
    cond_dir = Path(cond_dir)
    paths = sorted(cond_dir.glob("ep_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No ep_*.npz files under {cond_dir}")
    rows = [episode_metrics(load_episode(path)) for path in paths]
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _aggregate(per_ep: pd.DataFrame) -> dict[str, Any]:
    n = len(per_ep)
    t_success = per_ep.loc[per_ep["success"], "t_success_s"]
    return {
        "n_episodes": int(n),
        "success_rate": float(per_ep["success"].mean()),
        "grasp_rate": float(per_ep["grasped_any"].mean()),
        "lift_rate": float(per_ep["lifted"].mean()),
        "t_success_mean_s": float(t_success.mean()) if len(t_success) else float("nan"),
        "drop_after_lift_rate": float(per_ep["drop_after_lift"].mean()),
        "mass_kg": float(per_ep["mass_kg"].mean()),
    }


def metrics_dataframe(
    phase1_root: str | Path,
    cell: str,
    conditions: Iterable[str],
    model: str,
) -> pd.DataFrame:
    """Tidy per-condition metrics: one row per condition of ``cell``,
    with a ``model`` column, reading ``<phase1_root>/<cell>/<condition>/``.
    """
    phase1_root = Path(phase1_root)
    rows = []
    for condition in conditions:
        per_ep = condition_episode_frame(phase1_root / cell / condition)
        row: dict[str, Any] = {"model": model, "cell": cell, "condition": condition}
        row.update(_aggregate(per_ep))
        rows.append(row)
    columns = [
        "model",
        "cell",
        "condition",
        "mass_kg",
        "n_episodes",
        "success_rate",
        "grasp_rate",
        "lift_rate",
        "t_success_mean_s",
        "drop_after_lift_rate",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_csv(df: pd.DataFrame, csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase-1 metrics -> CSV.")
    parser.add_argument("--phase1-root", default="output/mass_variation/phase1")
    parser.add_argument("--cell", default="PickPlaceCounterToCabinet")
    parser.add_argument(
        "--conditions", nargs="+", default=["MassLight", "MassMedium", "MassHeavy"]
    )
    parser.add_argument("--model", default="xr1")
    parser.add_argument("--csv", default=None, help="Default: <phase1-root>/metrics_<model>.csv")
    args = parser.parse_args(argv)

    df = metrics_dataframe(args.phase1_root, args.cell, args.conditions, args.model)
    csv_path = (
        Path(args.csv)
        if args.csv
        else Path(args.phase1_root) / f"metrics_{args.model}.csv"
    )
    write_csv(df, csv_path)
    print(df.to_string(index=False))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

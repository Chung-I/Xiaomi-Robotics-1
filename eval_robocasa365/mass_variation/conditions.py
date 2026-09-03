# Copyright (C) 2026 Xiaomi Corporation.
"""Mass/CoM condition table for the XR1 mass/CoM study (Plan 1).

Pure module: no sim imports. Must import without robocasa installed.

Per Plan amendment A (docs/studies/2026-09-03-mass-com-xr1-plan-1-benchmark.md,
binding over the original Task 1 spec in the same doc): mass levels are fixed
constants -- there is no knee calibration and no ``load_mass_levels`` loader.
CoM conditions carry the "medium" mass level (not a knee ratio).
"""

from __future__ import annotations

CONDITIONS = ("MassLight", "MassMedium", "MassHeavy", "CoMOffA", "CoMOffB")

# Fixed mass levels in kg (Plan amendment A). No knee fitting in Phase 1.
MASS_LEVELS_KG = {"light": 0.15, "medium": 0.6, "heavy": 1.2}

# CoM offset axis, finalized per-cell by Task 2's settle gate; "y" is the
# default used throughout Task 1's preflight and the design doc.
_DEFAULT_COM_AXIS = "y"

_MASS_CONDITION_LEVEL = {
    "MassLight": "light",
    "MassMedium": "medium",
    "MassHeavy": "heavy",
}

_COM_CONDITION_SIGN = {
    "CoMOffA": 1.0,
    "CoMOffB": -1.0,
}


def condition_physics(
    condition: str,
    com_offset_m: float = 0.02,
    com_axis: str = _DEFAULT_COM_AXIS,
) -> dict:
    """Physics parameters for one condition.

    Mass conditions (MassLight/Medium/Heavy) return the corresponding
    MASS_LEVELS_KG value with zero CoM offset. CoM conditions (CoMOffA/B)
    return the "medium" mass level with +/-``com_offset_m`` on ``com_axis``.
    """
    if condition in _MASS_CONDITION_LEVEL:
        level = _MASS_CONDITION_LEVEL[condition]
        return {
            "mass_kg": MASS_LEVELS_KG[level],
            "com_offset_m": 0.0,
            "com_axis": com_axis,
        }

    if condition in _COM_CONDITION_SIGN:
        sign = _COM_CONDITION_SIGN[condition]
        return {
            "mass_kg": MASS_LEVELS_KG["medium"],
            "com_offset_m": sign * com_offset_m,
            "com_axis": com_axis,
        }

    raise ValueError(
        f"Unknown condition {condition!r}; expected one of {CONDITIONS}"
    )


def episode_seeds(base_seed: int, cell_index: int, n: int = 35) -> list[int]:
    """Matched-pair episode seeds for one cell.

    Formula per the plan's Global Constraints:
        seed_list = [base_seed + 1000 * cell_index + k for k in range(n)]

    Identical across all conditions of a cell by construction -- this
    function takes no condition argument.
    """
    return [base_seed + 1000 * cell_index + k for k in range(n)]

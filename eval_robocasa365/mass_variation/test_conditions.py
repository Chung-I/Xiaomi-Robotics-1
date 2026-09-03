# Copyright (C) 2026 Xiaomi Corporation.
"""Tests for the mass/CoM condition table (pure module, no sim imports).

Per Plan amendment A (docs/studies/2026-09-03-mass-com-xr1-plan-1-benchmark.md):
mass levels are fixed constants (no knee calibration / load_mass_levels), and
CoM conditions carry the "medium" mass level rather than a ratio of a runtime
knee value.
"""

import pytest

from eval_robocasa365.mass_variation.conditions import (
    CONDITIONS,
    MASS_LEVELS_KG,
    condition_physics,
    episode_seeds,
)


def test_conditions_tuple():
    assert CONDITIONS == (
        "MassLight",
        "MassMedium",
        "MassHeavy",
        "CoMOffA",
        "CoMOffB",
    )


def test_mass_levels_exact_kg():
    assert MASS_LEVELS_KG == {"light": 0.15, "medium": 0.6, "heavy": 1.2}

    light = condition_physics("MassLight")
    medium = condition_physics("MassMedium")
    heavy = condition_physics("MassHeavy")

    assert light["mass_kg"] == pytest.approx(0.15)
    assert medium["mass_kg"] == pytest.approx(0.6)
    assert heavy["mass_kg"] == pytest.approx(1.2)

    for physics in (light, medium, heavy):
        assert physics["com_offset_m"] == 0.0


def test_com_conditions_carry_matched_medium_mass():
    a = condition_physics("CoMOffA", com_offset_m=0.02)
    b = condition_physics("CoMOffB", com_offset_m=0.02)

    assert a["mass_kg"] == pytest.approx(MASS_LEVELS_KG["medium"])
    assert b["mass_kg"] == pytest.approx(MASS_LEVELS_KG["medium"])

    assert a["com_offset_m"] == pytest.approx(0.02)
    assert b["com_offset_m"] == pytest.approx(-0.02)

    assert a["com_axis"] == "y"
    assert b["com_axis"] == "y"


def test_condition_physics_default_com_offset():
    a = condition_physics("CoMOffA")
    assert a["com_offset_m"] == pytest.approx(0.02)


def test_condition_physics_unknown_condition_raises():
    with pytest.raises(ValueError):
        condition_physics("NotACondition")


def test_seeds_matched_across_conditions():
    assert episode_seeds(7, 1, 3) == [1007, 1008, 1009]


def test_seeds_default_n_is_35():
    seeds = episode_seeds(7, 0)
    assert len(seeds) == 35
    assert seeds[0] == 7
    assert seeds[-1] == 41


def test_seeds_identical_regardless_of_condition():
    # episode_seeds does not take a condition argument at all -- the same
    # seed list is reused across all conditions of a cell by construction.
    assert episode_seeds(7, 2, 5) == [2007, 2008, 2009, 2010, 2011]

# Copyright (C) 2026 Xiaomi Corporation.
"""Tests for the pure parts of the physics-injection module (overrides.py).

Only the parts that don't need a live MuJoCo/robocasa env: ``mass_to_density``
linearity, axis-index mapping, and idempotency of the authored-ipos
bookkeeping (exercised here with a fake numpy "model" -- no sim import).
Sim-touching behavior (install/uninstall against the real
robocasa.utils.env_utils.MJCFObject, apply_com_offset, settle_and_gate against
a real env) is exercised by verify_overrides.py, not here.
"""

import numpy as np
import pytest

from eval_robocasa365.mass_variation.overrides import (
    _authored_ipos,
    axis_index,
    mass_to_density,
)


def test_mass_to_density_matches_default_probe_density():
    # probe object weighs 0.06 kg at the default probe density (100) -> asking
    # for 0.6 kg (10x) must give 10x the density (1000).
    assert mass_to_density(0.6, 0.06) == pytest.approx(1000.0)


def test_mass_to_density_is_linear_in_target_mass():
    # ratio target/probe determines the density multiplier regardless of the
    # absolute probe mass -- verified 10x -> 10x property from the ground
    # truth exploration.
    for probe_mass in (0.01, 0.0268, 0.05453, 1.0):
        density = mass_to_density(10 * probe_mass, probe_mass, probe_density=100.0)
        assert density == pytest.approx(1000.0)


def test_mass_to_density_respects_probe_density_argument():
    # same target/probe mass ratio, different probe_density -> scales the
    # same way (probe_density is just the density that produced probe_mass).
    assert mass_to_density(0.6, 0.06, probe_density=50.0) == pytest.approx(500.0)


def test_mass_to_density_rejects_nonpositive_probe_mass():
    with pytest.raises(ValueError):
        mass_to_density(0.6, 0.0)
    with pytest.raises(ValueError):
        mass_to_density(0.6, -0.01)


def test_axis_index_mapping():
    assert axis_index("x") == 0
    assert axis_index("y") == 1
    assert axis_index("z") == 2


def test_axis_index_rejects_unknown_axis():
    with pytest.raises(ValueError):
        axis_index("w")


def test_authored_ipos_idempotent_across_repeated_writes():
    # Fake "model.body_ipos": a (n_bodies, 3) numpy array, standing in for
    # sim.model.body_ipos. First call for a given mesh_key caches a COPY of
    # the current row as "authored"; later calls -- even after the row has
    # been mutated in place (simulating a previously-applied offset) -- must
    # keep returning the ORIGINAL authored value, not the mutated one.
    body_ipos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.02, 0.03],
            [0.1, -0.2, 0.3],
        ]
    )
    cache: dict = {}
    bid = 1
    mesh_key = "milk_1"

    authored_first = _authored_ipos(cache, mesh_key, bid, body_ipos)
    assert np.allclose(authored_first, [0.01, 0.02, 0.03])

    # simulate a write (as apply_com_offset would perform) mutating the row.
    body_ipos[bid] = authored_first + np.array([0.0, 0.02, 0.0])

    authored_second = _authored_ipos(cache, mesh_key, bid, body_ipos)
    assert np.allclose(authored_second, [0.01, 0.02, 0.03])
    assert authored_second is not body_ipos[bid]


def test_authored_ipos_returns_a_copy_not_a_view():
    body_ipos = np.array([[1.0, 2.0, 3.0]])
    cache: dict = {}
    authored = _authored_ipos(cache, "mesh", 0, body_ipos)
    authored[0] = 999.0
    # mutating the returned array must not corrupt the cached authored value
    # or the underlying body_ipos row.
    assert cache["mesh"][0] == pytest.approx(1.0)
    assert body_ipos[0, 0] == pytest.approx(1.0)


def test_authored_ipos_different_mesh_keys_are_independent():
    body_ipos = np.array([[0.0, 0.0, 0.0], [5.0, 6.0, 7.0]])
    cache: dict = {}
    a = _authored_ipos(cache, "mesh_a", 0, body_ipos)
    b = _authored_ipos(cache, "mesh_b", 1, body_ipos)
    assert np.allclose(a, [0.0, 0.0, 0.0])
    assert np.allclose(b, [5.0, 6.0, 7.0])
    assert set(cache.keys()) == {"mesh_a", "mesh_b"}

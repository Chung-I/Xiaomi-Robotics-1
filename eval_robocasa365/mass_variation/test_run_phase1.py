# Copyright (C) 2026 Xiaomi Corporation.
"""Tests for the Phase-1 driver's pure, sim-free pieces (fix-wave item 4).

``run_phase1.py`` defers all heavy sim/model imports into ``main()``, so
this module -- ``require_sanity_gate_pass``, ``summary_path_for``,
``cell_dir_for`` -- imports and runs fine with only the stdlib + pandas
(via ``metrics``), no robocasa/torch/gymnasium needed.
"""

from __future__ import annotations

import json

import pytest

from eval_robocasa365.mass_variation import run_phase1


def _write_sanity_summary(output_root, model, passed, mass_light_success_rate=0.5, threshold=0.25):
    summary_path = run_phase1.summary_path_for(output_root, "sanity", model)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sanity_gate": {
            "passed": passed,
            "mass_light_success_rate": mass_light_success_rate,
            "threshold": threshold,
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file)
    return summary_path


class TestRequireSanityGatePass:
    def test_no_summary_yet_does_not_block(self, tmp_path):
        # Nothing to gate on -- allowed through, returns None.
        assert run_phase1.require_sanity_gate_pass(tmp_path, "xr1", force=False) is None

    def test_gate_passed_does_not_block(self, tmp_path):
        _write_sanity_summary(tmp_path, "xr1", passed=True)
        summary = run_phase1.require_sanity_gate_pass(tmp_path, "xr1", force=False)
        assert summary["sanity_gate"]["passed"] is True

    def test_gate_failed_blocks_by_default(self, tmp_path):
        _write_sanity_summary(tmp_path, "pi05_robocasa", passed=False, mass_light_success_rate=0.1)
        with pytest.raises(run_phase1.SanityGateBlocked, match="pi05_robocasa"):
            run_phase1.require_sanity_gate_pass(tmp_path, "pi05_robocasa", force=False)

    def test_gate_failed_with_force_proceeds(self, tmp_path):
        _write_sanity_summary(tmp_path, "pi05_robocasa", passed=False, mass_light_success_rate=0.1)
        summary = run_phase1.require_sanity_gate_pass(tmp_path, "pi05_robocasa", force=True)
        assert summary["sanity_gate"]["passed"] is False

    def test_only_the_named_models_summary_is_consulted(self, tmp_path):
        # xr1's FAILED sanity summary must not block a pi05_robocasa phase1
        # run -- the gate is per-model.
        _write_sanity_summary(tmp_path, "xr1", passed=False)
        assert run_phase1.require_sanity_gate_pass(tmp_path, "pi05_robocasa", force=False) is None


class TestPhase1ModeWiredToTheGate:
    def test_force_gate_flag_defaults_false(self):
        args = run_phase1.parse_args(["--mode", "phase1"])
        assert args.force_gate is False

    def test_force_gate_flag_parses(self):
        args = run_phase1.parse_args(["--mode", "phase1", "--force-gate"])
        assert args.force_gate is True

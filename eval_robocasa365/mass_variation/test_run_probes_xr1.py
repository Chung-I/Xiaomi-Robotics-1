# Copyright (C) 2026 Xiaomi Corporation.
"""Pure-helper tests for Plan-2 Task 4 (``analysis/run_probes_xr1.py``).

TDD scope (the plan's Task 4 Step 1 join + grid helpers -- everything here
runs without acts npz, GPU, or sim):

- ``join_index``: (episode_id, step) acts-row -> dataset-row index map, the
  seam where an off-by-one would silently misalign every label with its
  activation. Tested for exact alignment, the LAST valid step, out-of-range
  steps (loud), unknown episodes (loud), and duplicate dataset rows (loud).
- ``step_clock``: the per-episode ``step / T`` control target (T = that
  episode's total recorded steps, so the clock lives in [0, 1)).
- ``wrench_norm_of``: |F| from the (N, 3) ``ee_force`` channel.
- ``iter_feature_blocks``: the (block, layer, position) -> X enumeration
  over the capture contract's arrays (vlm / dit / state_embed), including
  the amendment-A layer-id mapping (array axis index vs real layer id).
- degenerate guards in ``probe_target_cell``: constant regression target,
  single-class classification target, too few groups for GroupKFold(5),
  and a class missing from some fold's training rows -- each must return a
  flagged degenerate cell, never a crash or a silent number.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_robocasa365.mass_variation.analysis.run_probes_xr1 import (
    DIT_POSITION_NAMES,
    VLM_POSITION_NAMES,
    iter_feature_blocks,
    join_index,
    probe_target_cell,
    step_clock,
    wrench_norm_of,
)


# ---------------------------------------------------------------------------
# join_index
# ---------------------------------------------------------------------------


def _toy_dataset(lengths: dict[str, int]):
    """Per-step dataset rows in load_study's layout: contiguous episodes,
    steps 0..T-1 ascending within each."""
    eids, steps = [], []
    for eid, T in lengths.items():
        eids.extend([eid] * T)
        steps.extend(range(T))
    return np.array(eids), np.array(steps, dtype=np.int64)


def test_join_index_exact_alignment():
    ds_eid, ds_step = _toy_dataset({"MassLight/ep_7": 5, "MassHeavy/ep_8": 7})
    acts_eid = np.array(
        ["MassLight/ep_7", "MassLight/ep_7", "MassHeavy/ep_8", "MassHeavy/ep_8"]
    )
    acts_step = np.array([0, 4, 3, 6], dtype=np.int64)
    idx = join_index(ds_eid, ds_step, acts_eid, acts_step)
    assert idx.shape == (4,)
    # the joined dataset rows carry EXACTLY the requested (episode, step)
    assert list(ds_eid[idx]) == list(acts_eid)
    assert list(ds_step[idx]) == list(acts_step)
    # explicit values: ep_7 starts at row 0, ep_8 at row 5
    assert list(idx) == [0, 4, 5 + 3, 5 + 6]


def test_join_index_last_step_ok_one_past_end_raises():
    """The off-by-one test: step T-1 is the last valid row; step T is not."""
    ds_eid, ds_step = _toy_dataset({"MassLight/ep_7": 469})
    idx = join_index(ds_eid, ds_step, np.array(["MassLight/ep_7"]), np.array([468]))
    assert list(idx) == [468]
    with pytest.raises(KeyError, match="469"):
        join_index(ds_eid, ds_step, np.array(["MassLight/ep_7"]), np.array([469]))


def test_join_index_unknown_episode_raises():
    ds_eid, ds_step = _toy_dataset({"MassLight/ep_7": 3})
    with pytest.raises(KeyError, match="ep_9"):
        join_index(ds_eid, ds_step, np.array(["MassLight/ep_9"]), np.array([0]))


def test_join_index_duplicate_dataset_row_raises():
    ds_eid = np.array(["MassLight/ep_7", "MassLight/ep_7"])
    ds_step = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate"):
        join_index(ds_eid, ds_step, np.array(["MassLight/ep_7"]), np.array([0]))


def test_join_index_replan_grid_shape():
    """A realistic replan grid (t = 0, 16, ...) over two episodes."""
    ds_eid, ds_step = _toy_dataset({"a/ep_1": 469, "b/ep_2": 33})
    grid_a = np.arange(0, 469, 16)
    grid_b = np.arange(0, 33, 16)  # 0, 16, 32 -- includes a step near T-1
    acts_eid = np.concatenate([np.repeat("a/ep_1", len(grid_a)), np.repeat("b/ep_2", len(grid_b))])
    acts_step = np.concatenate([grid_a, grid_b])
    idx = join_index(ds_eid, ds_step, acts_eid, acts_step)
    assert len(idx) == len(grid_a) + len(grid_b)
    assert list(ds_step[idx][-3:]) == [0, 16, 32]
    assert list(ds_step[idx][: len(grid_a)]) == list(grid_a)


# ---------------------------------------------------------------------------
# step_clock
# ---------------------------------------------------------------------------


def test_step_clock_basic():
    eid, step = _toy_dataset({"a/ep_1": 5})
    clock = step_clock(eid, step)
    np.testing.assert_allclose(clock, [0.0, 0.2, 0.4, 0.6, 0.8])


def test_step_clock_per_episode_T():
    """T is per-episode: the same absolute step maps to different clocks."""
    eid, step = _toy_dataset({"a/ep_1": 4, "b/ep_2": 8})
    clock = step_clock(eid, step)
    np.testing.assert_allclose(clock[:4], [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(clock[4:], np.arange(8) / 8.0)
    assert clock.max() < 1.0 and clock.min() == 0.0


def test_step_clock_on_subset_rows_uses_full_T():
    """Called on already-joined (replan-step) rows the clock must still be
    step/T of the FULL episode -- pass T explicitly for that case."""
    eid = np.array(["a/ep_1", "a/ep_1"])
    step = np.array([0, 32], dtype=np.int64)
    clock = step_clock(eid, step, total_steps={"a/ep_1": 64})
    np.testing.assert_allclose(clock, [0.0, 0.5])


# ---------------------------------------------------------------------------
# wrench_norm_of
# ---------------------------------------------------------------------------


def test_wrench_norm_of():
    ee_force = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]], dtype=np.float32)
    np.testing.assert_allclose(wrench_norm_of(ee_force), [5.0, 2.0])


def test_wrench_norm_of_rejects_bad_shape():
    with pytest.raises(ValueError):
        wrench_norm_of(np.zeros((4, 6)))


# ---------------------------------------------------------------------------
# iter_feature_blocks
# ---------------------------------------------------------------------------


def _toy_acts(m=6, vlm_layers=(0, 7), dit_layers=(0, 35), vh=5, dh=3):
    rng = np.random.default_rng(0)
    return {
        "vlm": rng.normal(size=(m, len(vlm_layers), 4, vh)).astype(np.float16),
        "vlm_layer_ids": tuple(vlm_layers),
        "dit": rng.normal(size=(m, len(dit_layers), 2, 2, dh)).astype(np.float16),
        "dit_layer_ids": tuple(dit_layers),
        "state_embed": rng.normal(size=(m, 4, dh)).astype(np.float16),
    }


def test_iter_feature_blocks_grid():
    acts = _toy_acts()
    blocks = list(iter_feature_blocks(acts))
    keys = [(b, l, p) for b, l, p, _ in blocks]
    # vlm: 2 layers x 4 positions; dit: 2 layers x (2 flow x 2 pos); state_embed: 1
    assert len(keys) == 2 * 4 + 2 * 4 + 1
    assert ("vlm", 7, VLM_POSITION_NAMES[0]) in keys
    assert ("vlm", 0, VLM_POSITION_NAMES[3]) in keys
    assert ("dit", 35, DIT_POSITION_NAMES[0]) in keys  # flow0:state_tokens_mean
    assert ("dit", 0, DIT_POSITION_NAMES[3]) in keys   # flow4:action_tokens_mean
    assert ("state_embed", -1, "state_tokens_flat") in keys


def test_iter_feature_blocks_slices_are_correct_and_f32():
    acts = _toy_acts()
    by_key = {(b, l, p): X for b, l, p, X in iter_feature_blocks(acts)}
    # vlm layer id 7 sits at array axis index 1; position 2 is camera 2's mean
    X = by_key[("vlm", 7, VLM_POSITION_NAMES[2])]
    assert X.dtype == np.float32 and X.shape == (6, 5)
    np.testing.assert_allclose(X, acts["vlm"][:, 1, 2, :].astype(np.float32))
    # dit layer id 0 (axis 0), flow step 4 (axis index 1), action mean (axis 1)
    X = by_key[("dit", 0, DIT_POSITION_NAMES[3])]
    assert X.shape == (6, 3)
    np.testing.assert_allclose(X, acts["dit"][:, 0, 1, 1, :].astype(np.float32))
    # state_embed is the flattened (4 x H) block
    X = by_key[("state_embed", -1, "state_tokens_flat")]
    assert X.shape == (6, 12)
    np.testing.assert_allclose(X, acts["state_embed"].reshape(6, -1).astype(np.float32))


# ---------------------------------------------------------------------------
# probe_target_cell degenerate guards
# ---------------------------------------------------------------------------


def _cell_inputs(n_groups=10, per=12, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * per, d)).astype(np.float32)
    cv_groups = np.repeat(np.arange(n_groups), per)
    # two episodes per cv group (a light and a heavy condition), like the corpus
    shuffle_groups = np.array(
        [f"g{g}/c{i % 2}" for g in range(n_groups) for i in range(per)]
    )
    return X, cv_groups, shuffle_groups, rng


def test_probe_target_cell_constant_reg_target_degenerate():
    X, cv, sh, _ = _cell_inputs()
    cell = probe_target_cell(X, np.zeros(len(X)), cv, sh, task="reg")
    assert cell["degenerate"] and "variance" in cell["degenerate_reason"]


def test_probe_target_cell_single_class_clf_degenerate():
    X, cv, sh, _ = _cell_inputs()
    cell = probe_target_cell(X, np.zeros(len(X), dtype=bool), cv, sh, task="clf")
    assert cell["degenerate"] and "class" in cell["degenerate_reason"]


def test_probe_target_cell_too_few_groups_degenerate():
    X, cv, sh, rng = _cell_inputs(n_groups=4)
    y = rng.normal(size=len(X))
    cell = probe_target_cell(X, y, cv, sh, task="reg")
    assert cell["degenerate"] and "group" in cell["degenerate_reason"]


def test_probe_target_cell_empty_mask_degenerate():
    cell = probe_target_cell(
        np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0), np.zeros(0), task="reg"
    )
    assert cell["degenerate"] and cell["n"] == 0


def test_probe_target_cell_real_signal_recovers():
    """End-to-end sanity on the non-degenerate path: a linear target is
    decoded (R^2 high, selectivity high), and rank_acc comes back for reg."""
    X, cv, sh, rng = _cell_inputs(n_groups=10, per=12, d=4)
    w = rng.normal(size=X.shape[1])
    y = X @ w + 0.05 * rng.normal(size=len(X))
    cell = probe_target_cell(X, y, cv, sh, task="reg")
    assert not cell["degenerate"]
    assert cell["real"] > 0.8
    assert cell["selectivity"] > 0.5
    assert 0.0 <= cell["rank_acc"] <= 1.0 and cell["rank_acc"] > 0.8
    assert cell["n"] == len(X) and cell["n_groups"] == 10


def test_probe_target_cell_clf_balanced_signal():
    X, cv, sh, rng = _cell_inputs()
    y = (X[:, 0] > 0).astype(np.int64)
    cell = probe_target_cell(X, y, cv, sh, task="clf")
    assert not cell["degenerate"]
    assert cell["real"] > 0.9  # balanced accuracy
    assert cell["rank_acc"] is None  # clf cells carry no rank_acc

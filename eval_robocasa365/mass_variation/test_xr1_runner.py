# Copyright (C) 2026 Xiaomi Corporation.
"""Pure (model-free, GPU-free) tests for the offline XR1 runner (Task 4).

Covers message/prompt parity with ``entry.EvalClient``, state padding parity,
the explicit-noise helpers, and the scoped ``torch.randn_like`` override.
The model-loading / CUDA path is exercised by the determinism gate
(``xr1_runner.py --gate``), not here.

Run: ``.venv-mibot/bin/python -m pytest eval_robocasa365/mass_variation/test_xr1_runner.py -v``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eval_robocasa365.entry import CAMERA_KEYS, EvalClient, center_crop
from eval_robocasa365.mass_variation.xr1_runner import (
    ACTION_CHUNK,
    DEFAULT_CHECKPOINT,
    MODEL_ACTION_DIM,
    NOISE_SHAPE,
    build_messages,
    make_noise,
    override_initial_noise,
    prepare_state,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = "pick the milk from the counter and place it in the cabinet"


def _fixture_images(seed: int = 0, frames: int = 4, size: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        key: rng.integers(0, 256, size=(frames, size, size, 3), dtype=np.uint8)
        for key in CAMERA_KEYS
    }


class _CropOnlyStub:
    """Standalone stand-in for EvalClient's self (only crop_ratio is touched)."""

    def __init__(self, crop_ratio: float) -> None:
        self.crop_ratio = crop_ratio


def _flatten_content(messages) -> list[dict]:
    return [item for message in messages for item in message["content"]]


def test_build_messages_parity_with_evalclient() -> None:
    """Byte-identical structure vs. EvalClient._build_messages on the same fixture."""
    images = _fixture_images()
    ours = build_messages(images, INSTRUCTION, crop_ratio=0.95)
    reference = EvalClient._build_messages(_CropOnlyStub(0.95), images, INSTRUCTION)

    assert [m["role"] for m in ours] == [m["role"] for m in reference] == ["user", "assistant"]
    ours_items = _flatten_content(ours)
    ref_items = _flatten_content(reference)
    assert len(ours_items) == len(ref_items)
    for ours_item, ref_item in zip(ours_items, ref_items):
        assert ours_item["type"] == ref_item["type"]
        if ours_item["type"] == "text":
            assert ours_item["text"] == ref_item["text"]
        else:  # video: list of PIL images -> compare pixels exactly
            ours_frames = [np.asarray(f) for f in ours_item["video"]]
            ref_frames = [np.asarray(f) for f in ref_item["video"]]
            assert len(ours_frames) == len(ref_frames)
            for a, b in zip(ours_frames, ref_frames):
                assert a.dtype == b.dtype and a.shape == b.shape
                assert np.array_equal(a, b)


def test_build_messages_exact_prompt_text() -> None:
    """The literal prompt strings the checkpoint was evaluated with."""
    messages = build_messages(_fixture_images(), INSTRUCTION, crop_ratio=0.95)
    texts = [item["text"] for item in _flatten_content(messages) if item["type"] == "text"]
    assert texts == [
        "Left camera: ",
        "\nRight camera: ",
        "\nWrist camera: ",
        f"\n\nGenerate robot actions for the task:\n{INSTRUCTION} /no_cot",
        "<cot></cot>",
    ]


def test_build_messages_applies_entry_center_crop() -> None:
    images = _fixture_images(seed=3)
    messages = build_messages(images, INSTRUCTION, crop_ratio=0.95)
    first_video = next(
        item for item in _flatten_content(messages) if item["type"] == "video"
    )
    expected = center_crop(images[CAMERA_KEYS[0]][0], 0.95)
    assert np.array_equal(np.asarray(first_video["video"][0]), np.asarray(expected))


def test_prepare_state_zero_pads_14_to_60() -> None:
    rng = np.random.default_rng(1)
    proprio = rng.normal(size=(4, 14)).astype(np.float32)
    state = prepare_state(proprio)
    assert state.shape == (1, 4, 60)
    assert state.dtype == np.float32
    assert np.array_equal(state[0, :, :14], proprio)
    assert np.all(state[0, :, 14:] == 0.0)


def test_prepare_state_casts_to_float32() -> None:
    state = prepare_state(np.ones((4, 14), dtype=np.float64))
    assert state.dtype == np.float32
    assert np.array_equal(state[0, :, :14], np.ones((4, 14), dtype=np.float32))


def test_noise_shape_matches_checkpoint_action_config() -> None:
    """NOISE_SHAPE must equal action_mask's shape: (1,) + std.shape from the
    checkpoint's action_config (processing_mibot.py:110-130)."""
    config = json.loads(
        (Path(DEFAULT_CHECKPOINT) / "preprocessor_config.json").read_text()
    )
    std = np.asarray(config["action_config"]["robocasa365"]["std"])
    assert std.shape == (ACTION_CHUNK, MODEL_ACTION_DIM)
    assert NOISE_SHAPE == (1, ACTION_CHUNK, MODEL_ACTION_DIM)


def test_make_noise_deterministic_and_seed_sensitive() -> None:
    a = make_noise(123)
    b = make_noise(123)
    c = make_noise(456)
    assert a.shape == NOISE_SHAPE and a.dtype == torch.float32
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_override_returns_injected_noise_and_restores() -> None:
    original = torch.randn_like
    noise = make_noise(7)
    ref = torch.zeros(NOISE_SHAPE, dtype=torch.float64)
    with override_initial_noise(noise):
        drawn = torch.randn_like(ref)
    assert torch.randn_like is original
    assert drawn.dtype == torch.float64
    assert torch.equal(drawn, noise.to(torch.float64))


def test_override_rejects_shape_mismatch() -> None:
    original = torch.randn_like
    with pytest.raises(RuntimeError, match="shape"):
        with override_initial_noise(make_noise(7)):
            torch.randn_like(torch.zeros(2, 3))
    assert torch.randn_like is original


def test_override_rejects_second_draw() -> None:
    original = torch.randn_like
    noise = make_noise(7)
    ref = torch.zeros(NOISE_SHAPE)
    with pytest.raises(RuntimeError, match="once"):
        with override_initial_noise(noise):
            torch.randn_like(ref)
            torch.randn_like(ref)
    assert torch.randn_like is original


def test_override_rejects_zero_draws() -> None:
    original = torch.randn_like
    with pytest.raises(RuntimeError, match="never"):
        with override_initial_noise(make_noise(7)):
            pass
    assert torch.randn_like is original


def test_override_restores_on_body_exception() -> None:
    original = torch.randn_like
    with pytest.raises(ValueError):
        with override_initial_noise(make_noise(7)):
            raise ValueError("boom")
    assert torch.randn_like is original

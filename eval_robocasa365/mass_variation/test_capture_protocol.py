# Copyright (C) 2026 Xiaomi Corporation.
"""Pure tests for Task 3 (Plan 2): capture protocol + window assembly.

No sim, no GPU, no model: these exercise the length-prefixed-pickle framing
(mirroring ``deploy/server.py``), the replan-step capture grid, the
per-(episode, step) noise derivation, the live-loop-identical observation
window assembly (via ``entry.sample_history`` imported as the reference),
the video-token span grouping the capture server uses to locate per-camera
image blocks, and the storage projection / abort logic.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from eval_robocasa365.entry import CAMERA_KEYS, sample_history
from eval_robocasa365.mass_variation.replay_capture import (
    ABORT_BYTES,
    NOISE_SHAPE,
    REPLAN_INTERVAL,
    ObsWindowBuilder,
    check_projected_bytes,
    noise_array,
    noise_seed_for,
    pack_message,
    per_step_bytes,
    projected_capture_bytes,
    recv_message,
    replan_steps,
    video_token_spans,
)


# ---------------------------------------------------------------------------
# Protocol framing
# ---------------------------------------------------------------------------


def _roundtrip(obj):
    """pack_message -> real socket -> recv_message (the exact production
    read path, not a re-implementation)."""
    a, b = socket.socketpair()
    try:
        a.sendall(pack_message(obj))
        return recv_message(b)
    finally:
        a.close()
        b.close()


def test_pack_unpack_roundtrip_nested_arrays():
    frames = {
        key: np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
        for key in CAMERA_KEYS
    }
    msg = {
        "type": "infer",
        "episode_id": "MassMedium/ep_20",
        "step": 32,
        "frames_by_cam": frames,
        "proprio_history": np.linspace(0, 1, 56, dtype=np.float32).reshape(4, 14),
        "instruction": "pick the milk from the counter and place it in the cabinet",
        "noise_seed": 2000032,
    }
    out = _roundtrip(msg)
    assert out.keys() == msg.keys()
    assert out["type"] == "infer" and out["step"] == 32
    assert out["noise_seed"] == 2000032
    for key in CAMERA_KEYS:
        np.testing.assert_array_equal(out["frames_by_cam"][key], frames[key])
        assert out["frames_by_cam"][key].dtype == np.uint8
    np.testing.assert_array_equal(out["proprio_history"], msg["proprio_history"])
    assert out["proprio_history"].dtype == np.float32
    assert out["instruction"] == msg["instruction"]


def test_pack_message_framing_matches_deploy_server():
    """First 4 bytes are a big-endian uint32 payload length (deploy/server.py
    unpacks ``struct.unpack(">I", ...)``), then exactly that many pickle
    bytes."""
    import pickle
    import struct

    payload = {"ok": True}
    raw = pack_message(payload)
    (length,) = struct.unpack(">I", raw[:4])
    assert length == len(raw) - 4
    assert pickle.loads(raw[4:]) == payload


def test_recv_message_on_closed_socket_returns_none():
    a, b = socket.socketpair()
    a.close()
    try:
        assert recv_message(b) is None
    finally:
        b.close()


def test_multiple_messages_in_sequence():
    a, b = socket.socketpair()
    try:
        a.sendall(pack_message({"i": 1}) + pack_message({"i": 2}))
        assert recv_message(b)["i"] == 1
        assert recv_message(b)["i"] == 2
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# Capture grid
# ---------------------------------------------------------------------------


def test_replan_steps_grid():
    assert REPLAN_INTERVAL == 16
    assert replan_steps(1) == [0]
    assert replan_steps(16) == [0]
    assert replan_steps(17) == [0, 16]
    assert replan_steps(361) == list(range(0, 361, 16))
    # count == ceil(T / 16): the policy replans whenever the plan empties.
    for total in (1, 15, 16, 17, 32, 33, 361, 700):
        assert len(replan_steps(total)) == -(-total // 16)


def test_replan_steps_rejects_nonpositive():
    with pytest.raises(ValueError):
        replan_steps(0)


# ---------------------------------------------------------------------------
# Noise derivation
# ---------------------------------------------------------------------------


def test_noise_seed_for_is_unique_per_episode_step():
    seen = set()
    for ep_seed in (7, 8, 41):
        for step in (0, 16, 640):
            seen.add(noise_seed_for(ep_seed, step))
    assert len(seen) == 9


def test_noise_array_deterministic_and_correct_shape():
    n1 = noise_array(noise_seed_for(7, 16))
    n2 = noise_array(noise_seed_for(7, 16))
    n3 = noise_array(noise_seed_for(7, 32))
    assert n1.shape == NOISE_SHAPE == (1, 16, 60)
    assert n1.dtype == np.float32
    np.testing.assert_array_equal(n1, n2)
    assert not np.array_equal(n1, n3)


def test_noise_array_matches_default_rng():
    """The plan pins the mechanism: ``default_rng(noise_seed)``."""
    seed = noise_seed_for(9, 48)
    expected = (
        np.random.default_rng(seed).standard_normal(NOISE_SHAPE).astype(np.float32)
    )
    np.testing.assert_array_equal(noise_array(seed), expected)


# ---------------------------------------------------------------------------
# Window assembly (reference: entry.sample_history on the live loop's deque)
# ---------------------------------------------------------------------------


def _synthetic_obs(t: int):
    """Frames/state whose contents encode the absolute obs index ``t``."""
    frames = {
        key: np.full((6, 6, 3), (t + ci) % 256, dtype=np.uint8)
        for ci, key in enumerate(CAMERA_KEYS)
    }
    state = np.full(14, float(t), dtype=np.float32)
    return frames, state


@pytest.mark.parametrize("t", [0, 1, 2, 5, 6, 7, 16, 33, 48])
def test_window_builder_matches_live_deque_semantics(t):
    """At replan step t the window must hold obs indices
    {t-6, t-4, t-2, t} left-clamped at 0 -- exactly what the live loop's
    bounded deque + ``sample_history`` produce."""
    import collections

    builder = ObsWindowBuilder()
    # Reference: the live loop's own structures, driven identically.
    queue_length = (4 - 1) * 2 + 1
    ref_state_queue: collections.deque = collections.deque(maxlen=queue_length)
    ref_image_queues = {
        key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS
    }

    for i in range(t + 1):
        frames, state = _synthetic_obs(i)
        builder.append(frames, state)
        ref_state_queue.append(state)
        for key in CAMERA_KEYS:
            ref_image_queues[key].append(frames[key])

    frames_win, proprio_win = builder.windows()

    expected_indices = [max(0, t - (4 - 1 - i) * 2) for i in range(4)]
    assert proprio_win.shape == (4, 14)
    np.testing.assert_array_equal(
        proprio_win[:, 0], np.array(expected_indices, dtype=np.float32)
    )
    # Bit-identical to the imported live-loop reference.
    np.testing.assert_array_equal(
        proprio_win, sample_history(ref_state_queue, 4, 2)
    )
    for key in CAMERA_KEYS:
        assert frames_win[key].shape == (4, 6, 6, 3)
        np.testing.assert_array_equal(
            frames_win[key], sample_history(ref_image_queues[key], 4, 2)
        )


def test_window_builder_empty_raises():
    with pytest.raises(ValueError):
        ObsWindowBuilder().windows()


# ---------------------------------------------------------------------------
# Video-token span grouping (per-camera image blocks in the tokenized prompt)
# ---------------------------------------------------------------------------


def test_video_token_spans_groups_runs_per_camera():
    # 3 cameras x 2 temporal blocks x 3 tokens each, separated by text (id 1).
    V = 151656
    ids = [9, 9]
    expected = []
    for _cam in range(3):
        cam_spans = []
        for _blk in range(2):
            ids += [1]
            start = len(ids)
            ids += [V, V, V]
            cam_spans.append((start, start + 3))
        expected.append(cam_spans)
        ids += [1, 1]
    ids += [9]
    spans = video_token_spans(np.array(ids), video_token_id=V, num_cameras=3)
    assert spans == expected


def test_video_token_spans_rejects_indivisible_runs():
    V = 151656
    ids = np.array([1, V, V, 1, V, 1])  # 2 runs, not divisible by 3 cameras
    with pytest.raises(ValueError):
        video_token_spans(ids, video_token_id=V, num_cameras=3)


def test_video_token_spans_rejects_no_runs():
    with pytest.raises(ValueError):
        video_token_spans(np.array([1, 2, 3]), video_token_id=151656, num_cameras=3)


# ---------------------------------------------------------------------------
# Storage projection
# ---------------------------------------------------------------------------


def test_per_step_bytes_exact():
    # f16 acts: VLM 36 layers x 4 positions x 2560; DiT 36 layers x 2 flow
    # steps x 2 positions x 1024; state_embed 4 x 1024; f32 actions 16 x 12.
    expected = (
        36 * 4 * 2560 * 2 + 36 * 2 * 2 * 1024 * 2 + 4 * 1024 * 2 + 16 * 12 * 4
    )
    assert per_step_bytes() == expected


def test_projected_bytes_and_abort_gate():
    # 105 episodes x ~23 replan steps: well under the 8 GB abort bar.
    total = projected_capture_bytes([23] * 105)
    assert total == 105 * 23 * per_step_bytes()
    assert total < ABORT_BYTES
    check_projected_bytes(total)  # must not raise
    with pytest.raises(RuntimeError):
        check_projected_bytes(ABORT_BYTES + 1)

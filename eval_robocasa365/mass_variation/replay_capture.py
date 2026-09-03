# Copyright (C) 2026 Xiaomi Corporation.
"""Task 3 (Plan 2, mass-com-xr1): bit-exact replay -> XR1 capture server client.

Replays every XR1 Phase-1 episode (the 105 npz under
``output/mass_variation/phase1/PickPlaceCounterToCabinet/Mass*/``) through a
FRESH env exactly the way ``extract_policy_state.py`` does (same
``setup_condition_physics`` intended-mass density re-derivation, same
measured-mass assert at ``atol=1e-9``, same strict success/liftoff/grasped/
obj_pos bit-exactness gate, never loosened), while rebuilding the LIVE
client's observation window at every replan step and posting it to the
capture server (``capture_server.py``, ``.venv-mibot``) on localhost:10087.

Window fidelity (the point of this file): the live evaluator keeps bounded
deques (maxlen 7 = (4-1)*2+1) of per-camera frames and 14-D proprio states,
seeded with the initial observation and appended after every env step, and
samples them with ``entry.sample_history(queue, 4, 2)`` whenever the action
plan empties (every 16 steps). ``ObsWindowBuilder`` below holds the SAME
deques and calls the SAME imported ``sample_history`` -- indices
{t-6, t-4, t-2, t} left-clamped at 0 -- so the frames/proprio the server
feeds ``XR1Runner.infer`` are bit-identical to what the live client saw.
Frames are sent RAW (256x256x3 uint8): the 0.95 center crop is applied
CLIENT-side in the live stack (``EvalClient._build_messages`` ->
``center_crop``), and ``XR1Runner.infer`` reuses exactly that path via
``xr1_runner.build_messages``, so the crop happens server-side here through
the very same code the live run used.

Protocol (mirrors ``deploy/server.py`` framing): 4-byte big-endian uint32
length prefix + pickle, both directions. Requests are dicts with a ``type``
key (``ping`` / ``infer`` / ``finalize`` / ``discard``); ``infer`` carries
{episode_id, step, frames_by_cam (3 cams x (4,256,256,3) uint8),
proprio_history (4,14) f32, instruction, noise_seed} and returns
{ok, actions (16,12) f32}. The server holds the hooked ``XR1Runner``,
buffers acts per episode, and writes
``output/mass_variation/activations/xr1/<cond>/ep_<seed>.npz`` on
``finalize`` (nothing on ``discard``).

Noise: fixed per (episode, step) -- ``noise_seed_for(seed, step)`` =
``seed * 100_000 + step`` (unique: steps < 100k), realized SERVER-side as
``np.random.default_rng(noise_seed).standard_normal((1, 16, 60)).astype(
np.float32)`` (the plan's pinned mechanism) and injected through
``xr1_runner.override_initial_noise``.

Instruction: read from the replayed env's own initial observation
(``annotation.human.task_description``) -- the Phase-1 npz do not record it,
but the replay reproduces the exact reset the live run performed (verified
by the measured-mass assert + the trace gate), and the instruction is a
deterministic function of that reset, so this IS the live instruction.

Determinism gate (run BEFORE the corpus): ``--gate`` captures one
(episode, step-set) TWICE, each pass against a freshly started server
process (exact-PID lifecycle), and requires bit-identical activation npz
bytes + returned actions across the two passes; verdict written to
``output/mass_variation/activations/determinism_gate.json``.

Run under the robocasa venv with EGL (the server runs under .venv-mibot and
is launched/killed by this driver):

    MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.replay_capture --gate
    MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.replay_capture [--time-budget-s S]

Resumable: episodes whose acts npz already exists (or that are recorded as
divergences in the manifest) are skipped, so bounded foreground invocations
can be chained until the corpus is complete.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import pickle
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval_robocasa365.entry import (
    CAMERA_KEYS,
    collect_images,
    observation_to_state,
    sample_history,
)
from eval_robocasa365.mass_variation.conditions import episode_seeds

log = logging.getLogger("replay_capture")

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- capture-grid / model constants (checkpoint facts, see xr1_runner.py's
# capture-site contract + config.json: vlm text 36 layers x 2560, dit 36
# layers x 1024, state_length 4, action chunk (16, 60) -> decoded (16, 12)).
REPLAN_INTERVAL = 16
OBS_HISTORY = 4
OBS_INTERVAL = 2
NOISE_SHAPE = (1, 16, 60)
VLM_LAYERS = 36
VLM_HIDDEN = 2560
VLM_POSITIONS = 4  # last_prefix_token + image_tokens_mean x 3 cameras
DIT_LAYERS = 36
DIT_FLOW_STEPS = (0, 4)
DIT_POSITIONS = 2  # state-block mean + action-block mean
DIT_HIDDEN = 1024
STATE_TOKENS = 4
ACTIONS_SHAPE = (16, 12)
ABORT_BYTES = 8 * 1024**3  # plan T3: abort if projected acts exceed 8 GB

DEFAULT_PORT = 10087
DEFAULT_HOST = "127.0.0.1"
DEFAULT_ACTS_ROOT = REPO_ROOT / "output" / "mass_variation" / "activations" / "xr1"
GATE_JSON_PATH = (
    REPO_ROOT / "output" / "mass_variation" / "activations" / "determinism_gate.json"
)
SERVER_PYTHON = REPO_ROOT / ".venv-mibot" / "bin" / "python"
MASS_CONDITIONS = ("MassLight", "MassMedium", "MassHeavy")
PHASE1_CELL = "PickPlaceCounterToCabinet"


# ---------------------------------------------------------------------------
# Pure protocol helpers (length-prefixed pickle, deploy/server.py framing)
# ---------------------------------------------------------------------------


def pack_message(obj: Any) -> bytes:
    """4-byte big-endian uint32 length prefix + pickle payload -- byte-level
    identical framing to ``deploy/server.py`` (struct format ``">I"``)."""
    payload = pickle.dumps(obj)
    return struct.pack(">I", len(payload)) + payload


def _recv_exact(sock: socket.socket, length: int) -> bytes | None:
    data = b""
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_message(sock: socket.socket) -> Any | None:
    """Read one length-prefixed pickle message; ``None`` on clean EOF."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return pickle.loads(payload)


# ---------------------------------------------------------------------------
# Pure capture-grid helpers
# ---------------------------------------------------------------------------


def replan_steps(total_steps: int) -> list[int]:
    """The policy's actual decision points in a ``total_steps``-long episode:
    the action plan empties at t = 0, 16, 32, ... (count = ceil(T/16))."""
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    return list(range(0, total_steps, REPLAN_INTERVAL))


def noise_seed_for(episode_seed: int, step: int) -> int:
    """Unique, stable noise seed per (episode, step). Steps are < 100_000
    (episode horizons are hundreds of steps), so no collisions."""
    return int(episode_seed) * 100_000 + int(step)


def noise_array(noise_seed: int) -> np.ndarray:
    """The plan-pinned noise mechanism: ``default_rng(noise_seed)`` ->
    float32 (1, 16, 60). The server converts this to a torch tensor and
    injects it via ``xr1_runner.override_initial_noise``."""
    return (
        np.random.default_rng(noise_seed)
        .standard_normal(NOISE_SHAPE)
        .astype(np.float32)
    )


class ObsWindowBuilder:
    """The live evaluator's observation history, replayed: same bounded
    deques (maxlen ``(4-1)*2+1``), same imported ``entry.sample_history``
    (indices {t-6, t-4, t-2, t}, left-clamped by the not-yet-full deque)."""

    def __init__(self, obs_history: int = OBS_HISTORY, obs_interval: int = OBS_INTERVAL):
        self.obs_history = obs_history
        self.obs_interval = obs_interval
        queue_length = (obs_history - 1) * obs_interval + 1
        self.image_queues: dict[str, collections.deque] = {
            key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS
        }
        self.state_queue: collections.deque = collections.deque(maxlen=queue_length)

    def append(self, frames_by_cam: dict[str, np.ndarray], state: np.ndarray) -> None:
        for key in CAMERA_KEYS:
            self.image_queues[key].append(frames_by_cam[key])
        self.state_queue.append(state)

    def windows(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        frames = {
            key: sample_history(queue, self.obs_history, self.obs_interval)
            for key, queue in self.image_queues.items()
        }
        proprio = sample_history(self.state_queue, self.obs_history, self.obs_interval)
        return frames, proprio


def video_token_spans(
    input_ids: np.ndarray, video_token_id: int, num_cameras: int = 3
) -> list[list[tuple[int, int]]]:
    """Group the contiguous runs of ``video_token_id`` in a tokenized prompt
    into ``num_cameras`` consecutive per-camera groups of ``(start, end)``
    half-open spans. The prompt places its three videos in camera order
    (left, right, wrist -- ``EvalClient._build_messages``), and the Qwen3VL
    processor expands each video into per-temporal-block token runs
    separated by timestamp text, so the run count must be a positive
    multiple of ``num_cameras`` with equal runs per camera."""
    ids = np.asarray(input_ids).reshape(-1)
    is_video = ids == video_token_id
    if not is_video.any():
        raise ValueError("video_token_spans: no video tokens in input_ids")
    padded = np.concatenate([[False], is_video, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    runs = list(zip(starts.tolist(), ends.tolist()))
    if len(runs) % num_cameras != 0:
        raise ValueError(
            f"video_token_spans: {len(runs)} video-token runs are not divisible "
            f"by {num_cameras} cameras -- prompt structure changed, refusing to guess"
        )
    per_cam = len(runs) // num_cameras
    return [runs[i * per_cam : (i + 1) * per_cam] for i in range(num_cameras)]


def per_step_bytes() -> int:
    """On-disk bytes per captured replan step (f16 acts + f32 actions)."""
    vlm = VLM_LAYERS * VLM_POSITIONS * VLM_HIDDEN * 2
    dit = DIT_LAYERS * len(DIT_FLOW_STEPS) * DIT_POSITIONS * DIT_HIDDEN * 2
    state_embed = STATE_TOKENS * DIT_HIDDEN * 2
    actions = ACTIONS_SHAPE[0] * ACTIONS_SHAPE[1] * 4
    return vlm + dit + state_embed + actions


def projected_capture_bytes(replan_counts: list[int]) -> int:
    return sum(int(c) for c in replan_counts) * per_step_bytes()


def check_projected_bytes(total_bytes: int) -> None:
    if total_bytes > ABORT_BYTES:
        raise RuntimeError(
            f"Projected activation storage {total_bytes / 1024**3:.2f} GB exceeds "
            f"the plan's {ABORT_BYTES / 1024**3:.0f} GB abort bar -- refusing to run."
        )


# ---------------------------------------------------------------------------
# Capture client (socket to capture_server.py)
# ---------------------------------------------------------------------------


class CaptureClient:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.sock = socket.create_connection((host, port), timeout=600.0)

    def request(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.sock.sendall(pack_message(msg))
        response = recv_message(self.sock)
        if response is None:
            raise ConnectionError("capture server closed the connection")
        if not response.get("ok", False):
            raise RuntimeError(
                f"capture server error for {msg.get('type')!r}: "
                f"{response.get('error')}\n{response.get('traceback', '')}"
            )
        return response

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Server lifecycle (fresh start, exact-PID kill; port helpers reused from
# run_phase1 -- imported lazily so this module stays light for the tests)
# ---------------------------------------------------------------------------


def start_capture_server(
    port: int,
    acts_root: Path,
    log_path: Path,
    checkpoint_dir: str | None = None,
    timeout_s: float = 600.0,
) -> subprocess.Popen:
    from eval_robocasa365.mass_variation.run_phase1 import _port_listening, _wait

    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(SERVER_PYTHON), "-u", "-m",
        "eval_robocasa365.mass_variation.capture_server",
        "--port", str(port), "--acts-root", str(acts_root),
    ]
    if checkpoint_dir:
        cmd += ["--checkpoint-dir", checkpoint_dir]
    env = dict(os.environ)
    env.setdefault("MIBOT_SERVER_SEED", "7")  # T6/Plan-1 serving convention
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    with open(log_path, "ab") as log_file:
        log_file.write(b"\n===== replay_capture server start =====\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
        )
    log.info("Launched capture server pid=%d; waiting for port %d ...", proc.pid, port)

    def _up() -> bool:
        if proc.poll() is not None:
            raise RuntimeError(
                f"capture server pid={proc.pid} exited early "
                f"(code {proc.returncode}); see {log_path}"
            )
        return _port_listening(port)

    if not _wait(_up, timeout_s):
        raise RuntimeError(f"capture server did not listen on {port} within {timeout_s}s")
    return proc


def stop_capture_server(proc: subprocess.Popen, port: int) -> None:
    """SIGTERM (then SIGKILL) the EXACT pid we launched; never pattern-kill."""
    import signal as _signal

    from eval_robocasa365.mass_variation.run_phase1 import _port_listening, _wait

    if proc.poll() is None:
        os.kill(proc.pid, _signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("capture server pid=%d survived SIGTERM; SIGKILL", proc.pid)
            os.kill(proc.pid, _signal.SIGKILL)
            proc.wait(timeout=30)
    _wait(lambda: not _port_listening(port), 30.0)


def ensure_port_free(port: int) -> None:
    """Refuse to start if something is already listening on the capture
    port -- per the ops rules this driver only ever kills pids it launched
    itself, so a pre-existing listener is an error to report, not to kill."""
    from eval_robocasa365.mass_variation.run_phase1 import server_pid

    pid = server_pid(port)
    if pid is not None:
        raise RuntimeError(
            f"port {port} is already in use by pid {pid}; stop it first "
            "(this driver only kills processes it launched itself)"
        )


# ---------------------------------------------------------------------------
# Episode replay + capture
# ---------------------------------------------------------------------------


def replay_capture_episode(
    npz_path: str | Path,
    client: CaptureClient,
    max_captures: int | None = None,
    obj_name: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Bit-exact replay one Phase-1 episode, posting the live-identical
    observation window to the capture server at every replan step; on a
    clean full replay (strict trace gate PASS) sends ``finalize`` so the
    server writes the acts npz; on divergence sends ``discard`` and raises
    ``ReplayDivergence`` (nothing written). ``max_captures`` limits the
    number of replan steps captured (determinism-gate mode); the trace gate
    only runs on full replays.
    """
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import convert_action

    from eval_robocasa365.mass_variation.entry_mass import PRIMARY_CATEGORY
    from eval_robocasa365.mass_variation.recorder import StepRecorder
    from eval_robocasa365.mass_variation.render_episode import (
        COM_OFFSET_ATOL_M,
        DEFAULT_OBJ_NAME,
        MASS_ATOL_KG,
        ReplayDivergence,
        _assert_close,
        parse_npz_path,
        setup_condition_physics,
    )

    obj_name = obj_name or DEFAULT_OBJ_NAME
    category = category or PRIMARY_CATEGORY
    npz_path = Path(npz_path)

    with np.load(npz_path) as data:
        npz = {key: data[key] for key in data.files}

    cell, condition, seed_from_path = parse_npz_path(npz_path)
    seed = int(npz["seed"])
    if seed != seed_from_path:
        raise ValueError(
            f"{npz_path}: seed mismatch -- path implies {seed_from_path}, "
            f"npz 'seed' scalar says {seed}"
        )
    episode_id = f"{condition}/ep_{seed}"

    recorded_actions = np.asarray(npz["actions"], dtype=np.float32)
    steps = recorded_actions.shape[0]
    capture_grid = replan_steps(steps)
    if max_captures is not None:
        capture_grid = capture_grid[:max_captures]
    capture_set = set(capture_grid)

    log.info(
        "Replay-capturing %s: cell=%s condition=%s seed=%d steps=%d captures=%d",
        npz_path, cell, condition, seed, steps, len(capture_grid),
    )

    # Default camera resolution (256x256), same as the live Phase-1 run --
    # NOT extract_policy_state's shrunk 128px cameras: the frames ARE the
    # payload here.
    env = gym.make(f"robocasa/{cell}", split="pretrain", obj_groups=category, seed=seed)
    infer_wall = 0.0
    captured: list[int] = []
    try:
        try:
            setup = setup_condition_physics(env, condition, seed, obj_name=obj_name)
            _assert_close(
                "mass_kg", setup["mass_kg"], float(npz["mass_kg"]), MASS_ATOL_KG, npz_path
            )
            _assert_close(
                "com_offset_m", setup["com_offset_m"], float(npz["com_offset_m"]),
                COM_OFFSET_ATOL_M, npz_path,
            )

            observation = setup["initial_observation"]
            instruction = str(observation["annotation.human.task_description"])

            builder = ObsWindowBuilder()
            builder.append(collect_images(observation), observation_to_state(observation))

            recorder = StepRecorder()
            info: dict[str, Any] = {}
            for t in range(steps):
                if t in capture_set:
                    frames_win, proprio_win = builder.windows()
                    tic = time.monotonic()
                    client.request({
                        "type": "infer",
                        "episode_id": episode_id,
                        "step": t,
                        "frames_by_cam": frames_win,
                        "proprio_history": proprio_win.astype(np.float32),
                        "instruction": instruction,
                        "noise_seed": noise_seed_for(seed, t),
                    })
                    infer_wall += time.monotonic() - tic
                    captured.append(t)
                    if max_captures is not None and len(captured) >= max_captures:
                        break

                observation, _, done, truncated, info = env.step(
                    convert_action(recorded_actions[t])
                )
                recorder.record(env, obj_name, recorded_actions[t])
                builder.append(
                    collect_images(observation), observation_to_state(observation)
                )
            replayed_success = bool(info.get("success", False))
        except Exception:
            # Leave nothing half-buffered on the server for this episode.
            try:
                client.request({"type": "discard", "episode_id": episode_id})
            except Exception:  # noqa: BLE001 -- server may be gone too
                pass
            raise
    finally:
        env.close()

    if max_captures is not None:
        # Gate mode: partial replay, no trace gate; finalize what we captured.
        response = client.request({
            "type": "finalize",
            "episode_id": episode_id,
            "expected_steps": captured,
            "scalars": {
                "seed": seed, "condition": condition, "instruction": instruction,
                "mass_kg": float(npz["mass_kg"]), "total_steps": steps,
                "partial": True,
            },
        })
        return {
            "episode_id": episode_id, "captured_steps": captured,
            "acts_path": response["acts_path"], "infer_wall_s": round(infer_wall, 2),
            "partial": True,
        }

    # Strict bit-exactness gate, identical to extract_policy_state.py's.
    divergences = _trace_divergences(recorder, npz, setup, seed, replayed_success, npz_path)
    if divergences:
        try:
            client.request({"type": "discard", "episode_id": episode_id})
        finally:
            pass
        raise ReplayDivergence(f"{npz_path}: replay diverged: " + "; ".join(divergences))

    response = client.request({
        "type": "finalize",
        "episode_id": episode_id,
        "expected_steps": captured,
        "scalars": {
            "seed": seed, "condition": condition, "instruction": instruction,
            "mass_kg": float(npz["mass_kg"]), "total_steps": steps,
            "success": replayed_success, "liftoff_step": int(npz["liftoff_step"]),
            "partial": False,
        },
    })
    return {
        "episode_id": episode_id,
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "captured_steps": len(captured),
        "acts_path": response["acts_path"],
        "acts_bytes": response["acts_bytes"],
        "infer_wall_s": round(infer_wall, 2),
        "success": replayed_success,
        "bit_exact": True,
    }


def _trace_divergences(
    recorder: Any,
    npz: dict[str, np.ndarray],
    setup: dict[str, Any],
    seed: int,
    replayed_success: bool,
    npz_path: Path,
) -> list[str]:
    """The render_episode/extract_policy_state strict gate, verbatim
    semantics: success + liftoff_step exact, grasped trace exact, obj_pos
    within a float32 round-trip tolerance."""
    import tempfile

    recorded_success = bool(npz["success"])
    recorded_liftoff = int(npz["liftoff_step"])
    recorded_grasped = np.asarray(npz["grasped"], dtype=bool)
    recorded_obj_pos = np.asarray(npz["obj_pos"], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        check_path = Path(tmp) / "replay_check.npz"
        recorder.finalize(
            check_path,
            mass_kg=setup["mass_kg"], com_offset_m=setup["com_offset_m"],
            com_axis=setup["com_axis"], seed=seed, success=replayed_success,
        )
        with np.load(check_path) as replay_data:
            replayed_liftoff = int(replay_data["liftoff_step"])
            replayed_grasped = np.asarray(replay_data["grasped"], dtype=bool)
            replayed_obj_pos = np.asarray(replay_data["obj_pos"], dtype=np.float32)

    divergences = []
    if replayed_success != recorded_success:
        divergences.append(
            f"success: replayed={replayed_success} recorded={recorded_success}"
        )
    if replayed_liftoff != recorded_liftoff:
        divergences.append(
            f"liftoff_step: replayed={replayed_liftoff} recorded={recorded_liftoff}"
        )
    if replayed_grasped.shape != recorded_grasped.shape or not np.array_equal(
        replayed_grasped, recorded_grasped
    ):
        divergences.append("grasped trace differs from the recording")
    if replayed_obj_pos.shape != recorded_obj_pos.shape or not np.allclose(
        replayed_obj_pos, recorded_obj_pos, atol=1e-4, rtol=0.0
    ):
        max_diff = (
            float(np.max(np.abs(replayed_obj_pos - recorded_obj_pos)))
            if replayed_obj_pos.shape == recorded_obj_pos.shape
            else float("nan")
        )
        divergences.append(
            f"obj_pos trace differs from the recording (max |diff|={max_diff:.3e} m)"
        )
    return divergences


# ---------------------------------------------------------------------------
# Determinism gate: one (episode, step-set) twice across two server restarts
# ---------------------------------------------------------------------------

GATE_EPISODE_CONDITION = "MassMedium"
GATE_EPISODE_SEED = 20
GATE_MAX_CAPTURES = 3  # replan steps {0, 16, 32}: exercises clamped + full windows


def run_determinism_gate(args: argparse.Namespace) -> int:
    phase1_root = Path(args.phase1_root)
    npz_path = (
        phase1_root / PHASE1_CELL / GATE_EPISODE_CONDITION
        / f"ep_{GATE_EPISODE_SEED}.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    gate_base = GATE_JSON_PATH.parent / "gate_runs"
    results = []
    for run_idx in (1, 2):
        acts_root = gate_base / f"run{run_idx}"
        log_path = gate_base / f"server_run{run_idx}.log"
        ensure_port_free(args.port)
        proc = start_capture_server(
            args.port, acts_root, log_path, checkpoint_dir=args.checkpoint_dir
        )
        try:
            client = CaptureClient(port=args.port)
            try:
                result = replay_capture_episode(
                    npz_path, client, max_captures=GATE_MAX_CAPTURES
                )
            finally:
                client.close()
        finally:
            stop_capture_server(proc, args.port)
        result["server_pid"] = proc.pid
        results.append(result)
        log.info("gate pass %d done: %s", run_idx, result)

    acts1 = _load_npz_dict(Path(results[0]["acts_path"]))
    acts2 = _load_npz_dict(Path(results[1]["acts_path"]))
    diffs: dict[str, Any] = {}
    bit_identical = True
    for key in sorted(set(acts1) | set(acts2)):
        if key not in acts1 or key not in acts2:
            diffs[key] = "missing in one run"
            bit_identical = False
            continue
        a, b = acts1[key], acts2[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            diffs[key] = f"shape/dtype mismatch: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
            bit_identical = False
        elif not np.array_equal(a, b):
            if np.issubdtype(a.dtype, np.number):
                diffs[key] = {
                    "max_abs_diff": float(
                        np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                    )
                }
            else:
                diffs[key] = "content differs"
            bit_identical = False
        else:
            diffs[key] = "bit-identical"

    meta1 = json.loads((Path(results[0]["acts_path"]).parents[1] / "meta.json").read_text())
    meta2 = json.loads((Path(results[1]["acts_path"]).parents[1] / "meta.json").read_text())
    meta_identical = meta1 == meta2

    verdict = "PASS" if (bit_identical and meta_identical) else "FAIL"
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "verdict": verdict,
        "bit_identical_acts_across_fresh_servers": bit_identical,
        "meta_json_identical": meta_identical,
        "per_key": diffs,
        "episode": f"{GATE_EPISODE_CONDITION}/ep_{GATE_EPISODE_SEED}",
        "captured_steps": results[0]["captured_steps"],
        "runs": results,
        "procedure": (
            "one episode replayed twice (bit-exact sim replay each time), each "
            "pass against a FRESH capture-server process (fresh model load, "
            "exact-PID lifecycle); all activation arrays, returned actions, and "
            "meta.json compared for bit-identity"
        ),
        "git_sha": git_sha,
    }
    GATE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict", "bit_identical_acts_across_fresh_servers", "meta_json_identical", "episode", "captured_steps")}, indent=2))
    print(f"[gate] verdict={verdict} -> {GATE_JSON_PATH}", flush=True)
    return 0 if verdict == "PASS" else 1


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Full-corpus driver (resumable, bounded, manifest, wandb)
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {"episodes": {}, "divergences": {}}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)


def build_pending(
    phase1_root: Path, acts_root: Path, seeds: list[int], manifest: dict[str, Any]
) -> list[tuple[str, int, Path, Path]]:
    pending = []
    for condition in MASS_CONDITIONS:
        for seed in seeds:
            key = f"{condition}/ep_{seed}"
            if key in manifest.get("divergences", {}):
                continue
            p1_path = phase1_root / PHASE1_CELL / condition / f"ep_{seed}.npz"
            if not p1_path.exists():
                continue
            acts_path = acts_root / condition / f"ep_{seed}.npz"
            if acts_path.exists():
                continue
            pending.append((condition, seed, p1_path, acts_path))
    return pending


def _replan_count(p1_path: Path) -> int:
    with np.load(p1_path) as data:
        total = int(data["actions"].shape[0])
    return math.ceil(total / REPLAN_INTERVAL)


def log_wandb(manifest: dict[str, Any], acts_root: Path, config: dict[str, Any]) -> str:
    import wandb

    episodes = manifest["episodes"]
    total_steps = sum(e["captured_steps"] for e in episodes.values())
    total_bytes = sum(e["acts_bytes"] for e in episodes.values())
    run = wandb.init(
        project="mass-com-xr1", job_type="capture", name="plan2-capture-xr1",
        config=config,
    )
    run.summary["episodes"] = len(episodes)
    run.summary["captured_steps"] = total_steps
    run.summary["acts_gb"] = total_bytes / 1024**3
    run.summary["divergences"] = len(manifest.get("divergences", {}))
    run.summary["mean_infer_wall_s_per_step"] = (
        sum(e["infer_wall_s"] for e in episodes.values()) / max(1, total_steps)
    )
    rows = [
        [e["episode_id"], e["condition"], e["seed"], e["steps"], e["captured_steps"],
         e["acts_bytes"], e["infer_wall_s"], e["success"]]
        for e in sorted(episodes.values(), key=lambda r: r["episode_id"])
    ]
    run.log({
        "episodes": wandb.Table(
            columns=["episode_id", "condition", "seed", "steps", "captured_steps",
                     "acts_bytes", "infer_wall_s", "success"],
            data=rows,
        )
    })
    url = run.url
    run.finish()
    return url


def run_capture(args: argparse.Namespace) -> int:
    phase1_root = Path(args.phase1_root)
    acts_root = Path(args.acts_root)
    manifest_path = acts_root / "capture_manifest.json"
    seeds = episode_seeds(args.base_seed, 0, args.n_seeds)

    manifest = load_manifest(manifest_path)
    pending = build_pending(phase1_root, acts_root, seeds, manifest)
    log.info("pending=%d (before --limit)", len(pending))
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        log.info("Nothing pending.")
        _maybe_finish(manifest, manifest_path, acts_root, args, phase1_root, seeds)
        return 0

    # Plan T3 budget rule: project the WHOLE corpus's activation bytes
    # in-code and abort before starting if over the 8 GB bar.
    all_counts = [
        _replan_count(phase1_root / PHASE1_CELL / cond / f"ep_{s}.npz")
        for cond in MASS_CONDITIONS
        for s in seeds
        if (phase1_root / PHASE1_CELL / cond / f"ep_{s}.npz").exists()
    ]
    projected = projected_capture_bytes(all_counts)
    check_projected_bytes(projected)
    log.info(
        "Projected corpus activation storage: %.2f GB over %d replan steps "
        "(per-step %.3f MB) -- under the %.0f GB abort bar.",
        projected / 1024**3, sum(all_counts), per_step_bytes() / 1024**2,
        ABORT_BYTES / 1024**3,
    )

    from eval_robocasa365.mass_variation.render_episode import ReplayDivergence

    ensure_port_free(args.port)
    server_log = acts_root / "capture_server.log"
    proc = start_capture_server(
        args.port, acts_root, server_log, checkpoint_dir=args.checkpoint_dir
    )
    t0 = time.monotonic()
    n_ok = n_div = 0
    try:
        client = CaptureClient(port=args.port)
        try:
            for condition, seed, p1_path, _acts_path in pending:
                if args.time_budget_s is not None and (
                    time.monotonic() - t0
                ) >= args.time_budget_s:
                    log.info("BUDGET_EXIT: time budget reached; re-invoke to resume.")
                    break
                key = f"{condition}/ep_{seed}"
                ep_t0 = time.monotonic()
                try:
                    result = replay_capture_episode(p1_path, client)
                    result["wall_s"] = round(time.monotonic() - ep_t0, 1)
                    manifest["episodes"][key] = result
                    n_ok += 1
                    log.info(
                        "OK %s: steps=%d captures=%d wall=%.1fs (infer %.1fs)",
                        key, result["steps"], result["captured_steps"],
                        result["wall_s"], result["infer_wall_s"],
                    )
                except ReplayDivergence as exc:
                    log.error("DIVERGED %s: %s", key, exc)
                    manifest["divergences"][key] = {
                        "condition": condition, "seed": seed,
                        "npz_path": str(p1_path), "error": str(exc),
                    }
                    n_div += 1
                save_manifest(manifest_path, manifest)
        finally:
            client.close()
    finally:
        stop_capture_server(proc, args.port)

    _maybe_finish(manifest, manifest_path, acts_root, args, phase1_root, seeds)
    remaining = build_pending(phase1_root, acts_root, seeds, manifest)
    summary = {
        "status": "complete" if not remaining else "incomplete",
        "captured_this_run": n_ok,
        "diverged_this_run": n_div,
        "remaining": len(remaining),
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _maybe_finish(
    manifest: dict[str, Any],
    manifest_path: Path,
    acts_root: Path,
    args: argparse.Namespace,
    phase1_root: Path,
    seeds: list[int],
) -> None:
    """When the corpus is complete and wandb has not been logged yet, log
    the run once and record its url in the manifest."""
    remaining = build_pending(phase1_root, acts_root, seeds, manifest)
    if remaining or args.no_wandb or manifest.get("wandb_url"):
        return
    if not manifest["episodes"]:
        return
    config = {
        "plan": "2026-09-03-mass-com-xr1-plan-2-probing task 3",
        "port": args.port,
        "acts_root": str(acts_root),
        "noise": "default_rng(seed*100000+step).standard_normal((1,16,60)) f32",
        "capture": {
            "vlm_layers": VLM_LAYERS, "vlm_positions": VLM_POSITIONS,
            "dit_layers": DIT_LAYERS, "dit_flow_steps": list(DIT_FLOW_STEPS),
            "dit_positions": DIT_POSITIONS, "dtype": "float16",
            "grid": f"replan steps every {REPLAN_INTERVAL}",
        },
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip(),
    }
    url = log_wandb(manifest, acts_root, config)
    manifest["wandb_url"] = url
    save_manifest(manifest_path, manifest)
    log.info("wandb url: %s", url)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 3 (Plan 2): bit-exact replay -> XR1 activation capture."
    )
    parser.add_argument("--phase1-root", default="output/mass_variation/phase1")
    parser.add_argument("--acts-root", default=str(DEFAULT_ACTS_ROOT))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--n-seeds", type=int, default=35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--time-budget-s", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--gate", action="store_true",
        help="run the two-server-restart determinism gate instead of the corpus",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.gate:
        return run_determinism_gate(args)
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())

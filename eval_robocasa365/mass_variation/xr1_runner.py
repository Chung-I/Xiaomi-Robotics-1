# Copyright (C) 2026 Xiaomi Corporation.
"""Offline in-process XR1 runner + determinism gate (Plan 1, Task 4).

Loads ``MiBoTForActionGeneration`` directly (no pickle server) and exposes
``XR1Runner.infer(frames_by_cam, proprio_history, instruction, noise=None)``
with an EXPLICIT flow-matching noise override, so later plans can hold the
diffusion noise fixed while varying hidden physics. Prompt/state building is
imported from ``eval_robocasa365.entry`` (``EvalClient._build_messages``; the
state-padding lines of ``EvalClient.infer``), never copied.

Investigation notes (checkpoint code is READ-ONLY; all line numbers refer to
``checkpoints/Xiaomi-Robotics-1-RoboCasa365/modeling_mibot.py`` unless noted)
-----------------------------------------------------------------------------
Sampler seam:
  ``MiBoTForActionGeneration.forward(self, state, action_mask, num_steps=5,
  **kwargs)`` -- :1820-1874 (``@torch.no_grad()``). ``**kwargs`` goes verbatim
  to ``self.vlm(**kwargs, use_cache=True)`` (:1828). The flow-matching start
  noise enters at exactly ONE call site:

      :1866  ``x = torch.randn_like(action_mask)``

  followed by a 5-step Euler integration (:1867-1872). The only other
  ``randn`` in the file is an ``__init__``-time parameter init (:1701), so
  during ``forward`` the process-wide ``torch.randn_like`` is hit exactly
  once, with ``action_mask`` (shape ``(1, 16, 60)``, bf16, cuda) as ref.

Noise-injection mechanism chosen: scoped ``torch.randn_like`` patch
(``override_initial_noise`` below) around the single ``model(**inputs)``
call. Rationale vs. the subclass/``types.MethodType`` route the brief
prefers by default: overriding ``forward`` cannot inject ONLY the noise --
the draw sits mid-function, so an override must re-implement the whole
55-line body (position-id arithmetic :1836-1841, mask assembly :1844-1847,
Euler loop), i.e. copy checkpoint code that then silently drifts if the
checkpoint changes. The scoped patch duplicates nothing, is active only
inside the one call in THIS process (the pickle server is untouched), and is
made strict: it verifies ref shape == noise shape and that exactly one draw
happened, so any checkpoint change that adds/moves a ``randn_like`` fails
loudly instead of corrupting an experiment. ``noise=None`` leaves torch
completely untouched (identical to server behavior, drawing from the global
RNG stream seeded at import by ``MIBOT_SERVER_SEED``, :29-41).

Capture-site contract for Plan 2 (hook points, verified in this checkpoint):
  - VLM decoder layers: ``model.vlm.model.language_model.layers[i]``, i in
    0..35 -- ``Qwen3VLTextModel`` :762 builds ``self.layers`` :772-774
    (layer class ``Qwen3VLTextDecoderLayer`` :488); ``Qwen3VLModel``
    :884 holds ``self.language_model`` :895; ``Qwen3VLForConditionalGeneration``
    :1262 holds ``self.model`` :1271. One forward per inference (KV cached).
  - DiT layers: ``model.dit.layers[i]``, i in 0..35 -- ``DiT`` :1720 builds
    ``self.layers`` :1727-1729 (layer class ``DecoderLayer`` :1690, forward
    :1703-1718). Each fires ``num_steps=5`` times per inference (Euler loop
    :1869-1872 -> ``dit_forward`` :1798-1818 -> ``DiT.forward`` :1732-1742).
  - VLM->DiT KV coupling: DiT layer i consumes VLM layer ``start_index + i``
    KV (``start_index = len(past_key_values) - len(self.layers)`` :1733;
    36 == 36 here, so i <-> i). The cache is concatenated in front of the
    DiT tokens inside ``Attention.forward`` :1642-1673 (k/v concat
    :1661-1662, SDPA :1664-1670).
  - State embedding: ``state_embed = self.state_projector(state)`` :1850
    (``self.state_projector`` :1763-1768); hook ``model.state_projector``.
    DiT token layout per step: ``[sink(1), state(4), noisy_action(16)]``
    (:1808-1809), so DiT sequence length is 21.
  - Action head in/out: ``model.action_projector`` (:1769) input at :1805,
    ``model.action_output_layer`` (:1775) output at :1816.

Serving parity (so offline == server modulo noise):
  - ``deploy/server.py:59-61``: tensors moved to model device; floating
    tensors cast to model dtype (bf16); ``task_id`` stays in the request and
    flows into ``**kwargs``. Mirrored in ``XR1Runner.infer``.
  - ``deploy/client.py:61``: ``processor.decode_action(actions_cpu,
    robot_type)`` (un-normalize, ``processing_mibot.py:132-150``); then
    ``entry.py:184`` slices ``[0, :, :12]`` -> float32 ``(16, 12)``.
  - ``action_mask`` = ``(std > 1e-5)`` from the checkpoint action config,
    shape ``(1, 16, 60)`` (``processing_mibot.py:110-130``; mean/std are
    ``(16, 60)`` in ``preprocessor_config.json``).

Determinism gate (Step 3): ``--gate`` spawns TWO FRESH processes of
``--gate-worker``, each building the same synthetic observation fixture and
saving actions for noise seeds 123 (twice) and 456 plus a free-running
(noise=None) call; the parent asserts cross-process bit-identity for equal
noise and difference for different noise, writing
``output/mass_variation/determinism_gate.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from eval_robocasa365.entry import (
    ACTION_DIM,
    CAMERA_KEYS,
    ROBOT_TYPE,
    STATE_DIM,
    EvalClient,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "Xiaomi-Robotics-1-RoboCasa365"
GATE_DIR = REPO_ROOT / "output" / "mass_variation"

# action_mask shape = (1,) + std.shape, std is (16, 60): processing_mibot.py:110-130.
ACTION_CHUNK = 16
MODEL_ACTION_DIM = 60
NOISE_SHAPE = (1, ACTION_CHUNK, MODEL_ACTION_DIM)


# ---------------------------------------------------------------------------
# Pure helpers (no model, no GPU)
# ---------------------------------------------------------------------------

class _CropRatioCarrier:
    """Duck-typed ``self`` for ``EvalClient._build_messages`` (which only
    reads ``self.crop_ratio``), so the exact eval prompt builder is reused
    by import without opening a socket (EvalClient.__init__ connects)."""

    def __init__(self, crop_ratio: float) -> None:
        self.crop_ratio = crop_ratio


def build_messages(
    frames_by_cam: dict[str, Any], instruction: str, crop_ratio: float = 0.95
) -> list[dict[str, Any]]:
    """Byte-identical prompt structure to the stock evaluator (entry.py:134-159)."""
    return EvalClient._build_messages(_CropRatioCarrier(crop_ratio), frames_by_cam, instruction)


def prepare_state(proprio_history: np.ndarray) -> np.ndarray:
    """Zero-pad the (T, 14) proprio history to (1, T, 60), mirroring
    ``EvalClient.infer`` (entry.py:167-169) exactly."""
    proprio_history = np.asarray(proprio_history, dtype=np.float32)
    state = np.zeros((1, proprio_history.shape[0], STATE_DIM), dtype=np.float32)
    state[0, :, : proprio_history.shape[-1]] = proprio_history
    return state


def make_noise(seed: int, shape: tuple[int, ...] = NOISE_SHAPE) -> torch.Tensor:
    """Reproducible float32 CPU noise from a local generator (global RNG untouched)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float32)


@contextlib.contextmanager
def override_initial_noise(noise: torch.Tensor) -> Iterator[list[tuple[int, ...]]]:
    """Scoped, strict replacement of ``torch.randn_like`` for one model call.

    The sampler draws its start noise exactly once (modeling_mibot.py:1866,
    the only ``randn_like`` in the checkpoint). This context returns ``noise``
    (cast to the ref tensor's device/dtype) for that draw and:
      - raises if the ref shape differs from ``noise`` (seam moved),
      - raises if drawn more than once (a second sampling site appeared),
      - raises if never drawn (the patched call was bypassed),
      - always restores ``torch.randn_like``, including on exceptions.
    """
    original = torch.randn_like
    calls: list[tuple[int, ...]] = []

    def patched(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if tuple(input.shape) != tuple(noise.shape):
            raise RuntimeError(
                f"override_initial_noise: randn_like ref shape {tuple(input.shape)} "
                f"!= injected noise shape {tuple(noise.shape)}; the sampling seam "
                "(modeling_mibot.py:1866) has moved -- refusing to guess."
            )
        calls.append(tuple(input.shape))
        if len(calls) > 1:
            raise RuntimeError(
                "override_initial_noise: torch.randn_like drawn more than once "
                "inside the override scope; expected exactly one draw "
                "(modeling_mibot.py:1866)."
            )
        return noise.to(device=input.device, dtype=input.dtype)

    torch.randn_like = patched
    try:
        yield calls
        if not calls:
            raise RuntimeError(
                "override_initial_noise: torch.randn_like was never drawn inside "
                "the override scope; the injected noise was NOT used."
            )
    finally:
        torch.randn_like = original


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class XR1Runner:
    """In-process XR1 (MiBoT) policy with optional explicit start noise."""

    def __init__(
        self,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        crop_ratio: float = 0.95,
        robot_type: str = ROBOT_TYPE,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        checkpoint_dir = str(checkpoint_dir)
        # Mirrors deploy/client.py:22 (processor) and deploy/server.py:20 (model).
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_dir, trust_remote_code=True, use_fast=False
        )
        robot_types = self.processor.list_robot_types()
        if robot_type not in robot_types:
            raise ValueError(
                f"Robot type {robot_type!r} missing from checkpoint; available: {robot_types}"
            )
        self.model = (
            AutoModel.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        self.model.eval()
        self.device = device
        self.crop_ratio = crop_ratio
        self.robot_type = robot_type

    def infer(
        self,
        frames_by_cam: dict[str, Any],
        proprio_history: np.ndarray,
        instruction: str,
        noise: torch.Tensor | None = None,
    ) -> np.ndarray:
        """One inference -> float32 (16, 12) action chunk.

        ``frames_by_cam`` maps each of entry.CAMERA_KEYS to the sampled image
        history (sequence of HxWx3 uint8 frames, newest last), exactly what
        ``entry.sample_history`` produces. ``proprio_history`` is the sampled
        (T, 14) state history from ``entry.observation_to_state``. ``noise``,
        if given, replaces the flow-matching start noise (shape NOISE_SHAPE).
        """
        missing = [key for key in CAMERA_KEYS if key not in frames_by_cam]
        if missing:
            raise KeyError(f"frames_by_cam missing camera keys: {missing}")
        if noise is not None and tuple(noise.shape) != NOISE_SHAPE:
            raise ValueError(f"noise must have shape {NOISE_SHAPE}, got {tuple(noise.shape)}")

        state = prepare_state(proprio_history)
        inputs = self.processor.apply_chat_template(
            build_messages(frames_by_cam, instruction, self.crop_ratio),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=state,
            robot_type=self.robot_type,
        )
        request = dict(inputs)
        request["task_id"] = self.robot_type

        # Device/dtype move mirrors deploy/server.py:59 exactly (task_id, a
        # str, passes through and lands in the VLM **kwargs, same as serving).
        request = {
            key: (
                value.to(device=self.model.device, dtype=self.model.dtype)
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value.to(device=self.model.device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in request.items()
        }

        if noise is not None:
            with override_initial_noise(noise):
                outputs = self.model(**request)
        else:
            outputs = self.model(**request)

        # Decode mirrors deploy/client.py:61 + entry.py:184.
        actions = self.processor.decode_action(outputs.actions.cpu(), robot_type=self.robot_type)
        actions = actions[0, :, :ACTION_DIM].float().cpu().numpy()
        return np.asarray(actions, dtype=np.float32)


# ---------------------------------------------------------------------------
# Determinism gate
# ---------------------------------------------------------------------------

GATE_NOISE_SEED = 123
GATE_ALT_NOISE_SEED = 456
GATE_INSTRUCTION = "pick the milk from the counter and place it in the cabinet"


def build_gate_fixture(seed: int = 0, size: int = 256, frames: int = 4):
    """Deterministic synthetic observation (identical across processes)."""
    rng = np.random.default_rng(seed)
    frames_by_cam = {
        key: rng.integers(0, 256, size=(frames, size, size, 3), dtype=np.uint8)
        for key in CAMERA_KEYS
    }
    proprio = (rng.standard_normal((frames, 14)) * 0.1).astype(np.float32)
    return frames_by_cam, proprio


def run_gate_worker(out_path: Path, checkpoint_dir: str | Path, device: str) -> None:
    frames_by_cam, proprio = build_gate_fixture()
    runner = XR1Runner(checkpoint_dir=checkpoint_dir, device=device)
    actions_seed123 = runner.infer(
        frames_by_cam, proprio, GATE_INSTRUCTION, noise=make_noise(GATE_NOISE_SEED)
    )
    actions_seed123_repeat = runner.infer(
        frames_by_cam, proprio, GATE_INSTRUCTION, noise=make_noise(GATE_NOISE_SEED)
    )
    actions_seed456 = runner.infer(
        frames_by_cam, proprio, GATE_INSTRUCTION, noise=make_noise(GATE_ALT_NOISE_SEED)
    )
    actions_free = runner.infer(frames_by_cam, proprio, GATE_INSTRUCTION, noise=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        actions_seed123=actions_seed123,
        actions_seed123_repeat=actions_seed123_repeat,
        actions_seed456=actions_seed456,
        actions_free=actions_free,
    )
    print(f"[gate-worker] wrote {out_path}", flush=True)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def run_gate(checkpoint_dir: str | Path, device: str) -> int:
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    npz_paths = [GATE_DIR / f"determinism_gate_run{i}.npz" for i in (1, 2)]
    env = dict(os.environ)
    env.setdefault("MIBOT_SERVER_SEED", "7")
    for npz_path in npz_paths:  # TWO FRESH PROCESSES, sequential (one GPU)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval_robocasa365.mass_variation.xr1_runner",
                "--gate-worker",
                str(npz_path),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--device",
                device,
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )

    run1, run2 = (np.load(p) for p in npz_paths)
    a1, a2 = run1["actions_seed123"], run2["actions_seed123"]
    same_noise_bit_identical = bool(np.array_equal(a1, a2))
    same_noise_max_abs_diff = _max_abs_diff(a1, a2)
    in_process_repeat_identical = bool(
        np.array_equal(run1["actions_seed123"], run1["actions_seed123_repeat"])
    ) and bool(np.array_equal(run2["actions_seed123"], run2["actions_seed123_repeat"]))
    diff_noise_max_abs_diff = _max_abs_diff(run1["actions_seed123"], run1["actions_seed456"])
    different_noise_differs = not np.array_equal(
        run1["actions_seed123"], run1["actions_seed456"]
    )

    if same_noise_bit_identical:
        verdict = "PASS"
    elif same_noise_max_abs_diff < 1e-6:
        verdict = "PASS_WITH_TOLERANCE"  # documented per brief; proceed
    else:
        verdict = "BLOCKED"
    if not different_noise_differs:
        verdict = "BLOCKED"

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "verdict": verdict,
        "same_noise_bit_identical_across_fresh_processes": same_noise_bit_identical,
        "same_noise_max_abs_diff": same_noise_max_abs_diff,
        "in_process_repeat_bit_identical": in_process_repeat_identical,
        "different_noise_differs": different_noise_differs,
        "different_noise_max_abs_diff": diff_noise_max_abs_diff,
        "actions_shape": list(a1.shape),
        "noise_seeds": {"same": GATE_NOISE_SEED, "different": GATE_ALT_NOISE_SEED},
        "noise_shape": list(NOISE_SHAPE),
        "mechanism": (
            "scoped torch.randn_like override around the single model call "
            "(override_initial_noise); intercepts the one start-noise draw at "
            "modeling_mibot.py:1866, strict shape/count checks, restored in finally; "
            "checkpoint code unedited, server path untouched"
        ),
        "fixture": "build_gate_fixture(seed=0): 3 cams x 4 frames x 256x256x3 uint8, proprio (4,14) float32",
        "instruction": GATE_INSTRUCTION,
        "checkpoint_dir": str(checkpoint_dir),
        "env": {
            "MIBOT_SERVER_SEED": env["MIBOT_SERVER_SEED"],
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "git_sha": git_sha,
        "npz_runs": [str(p) for p in npz_paths],
    }
    gate_path = GATE_DIR / "determinism_gate.json"
    gate_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[gate] verdict={verdict} -> {gate_path}", flush=True)
    return 0 if verdict in ("PASS", "PASS_WITH_TOLERANCE") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gate", action="store_true", help="run the two-process determinism gate")
    group.add_argument("--gate-worker", metavar="OUT_NPZ", help="single-process gate worker")
    args = parser.parse_args(argv)

    if args.gate_worker:
        run_gate_worker(Path(args.gate_worker), args.checkpoint_dir, args.device)
        return 0
    return run_gate(args.checkpoint_dir, args.device)


if __name__ == "__main__":
    raise SystemExit(main())

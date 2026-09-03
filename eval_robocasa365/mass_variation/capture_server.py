# Copyright (C) 2026 Xiaomi Corporation.
"""Task 3 (Plan 2, mass-com-xr1): XR1 activation capture server (.venv-mibot).

Holds a hooked in-process ``XR1Runner`` behind the same length-prefixed
pickle framing as ``deploy/server.py`` (see ``replay_capture.py`` for the
message schema) and captures, per ``infer`` request:

- ALL 36 VLM decoder layers (``model.vlm.model.language_model.layers[i]``,
  the Plan-1 T4 capture contract; plan amendment A: capture all 36, probing
  later samples {0,7,14,21,28,35}) at 4 prefix positions:
  ``last_prefix_token`` plus ``image_tokens_mean`` for each of the 3
  cameras (mean over that camera's video-token spans, located from
  ``input_ids`` via ``replay_capture.video_token_spans``; the 3 videos
  precede the instruction text in the prompt, so the spans are constant
  across episodes -- asserted on every request).
- ALL 36 DiT layers (``model.dit.layers[j]``) at flow steps {0, 4} of the
  5-step Euler loop (disambiguated by a per-layer call counter; layer j's
  n-th firing IS flow step n), at 2 positions of the 21-token DiT sequence
  ``[sink(1), state(4), noisy_action(16)]``: state-block mean and
  action-block mean.
- ``state_embed``: the ``model.state_projector`` output, (4, 1024). NOTE on
  the contract's 'state_token' position: this checkpoint's VLM prefix
  carries NO state token (state enters the DiT sequence via
  ``state_projector``, modeling_mibot.py:1850), so the position is served
  by this hook plus the per-DiT-layer state-block means, and the VLM
  position set is the 4 above. Recorded in meta.json.

Fire-count asserts per inference (the T4-contract pattern -- any checkpoint
change that alters the call graph fails loudly instead of corrupting the
capture): every VLM layer hook exactly once (single prefix pass, KV
cached), every DiT layer hook exactly num_steps=5 times, ``state_projector``
exactly once, the ``input_ids`` pre-hook exactly once.

Noise: realized server-side from the request's ``noise_seed`` via
``replay_capture.noise_array`` (``default_rng`` mechanism pinned by the
plan) and injected through ``xr1_runner.override_initial_noise`` -- the
strict scoped ``randn_like`` patch (single-draw + shape asserts).

Acts are buffered per episode (f16, computed as f32 means then cast) and
written to ``<acts-root>/<cond>/ep_<seed>.npz`` on ``finalize`` (tmp file +
atomic rename); ``discard`` drops the buffer. ``meta.json`` (layer names,
positions, token-block spans, git SHA, versions, determinism note) is
written next to the condition dirs on the first inference and its
span-derived content re-asserted on every later one.

Launch (normally done by ``replay_capture.py``; exact-PID lifecycle):

    MIBOT_SERVER_SEED=7 .venv-mibot/bin/python -u -m \\
        eval_robocasa365.mass_variation.capture_server --port 10087 \\
        --acts-root output/mass_variation/activations/xr1
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_robocasa365.entry import CAMERA_KEYS
from eval_robocasa365.mass_variation.replay_capture import (
    ACTIONS_SHAPE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DIT_FLOW_STEPS,
    DIT_HIDDEN,
    DIT_LAYERS,
    STATE_TOKENS,
    VLM_HIDDEN,
    VLM_LAYERS,
    noise_array,
    pack_message,
    recv_message,
    video_token_spans,
)
from eval_robocasa365.mass_variation.xr1_runner import (
    DEFAULT_CHECKPOINT,
    XR1Runner,
    override_initial_noise,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DIT_NUM_STEPS = 5  # modeling_mibot.py forward(num_steps=5) Euler loop
DIT_SEQ_LEN = 1 + STATE_TOKENS + 16  # [sink(1), state(4), noisy_action(16)]
DIT_STATE_SLICE = slice(1, 1 + STATE_TOKENS)
DIT_ACTION_SLICE = slice(1 + STATE_TOKENS, DIT_SEQ_LEN)

VLM_POSITION_NAMES = ["last_prefix_token"] + [
    f"image_tokens_mean:{key}" for key in CAMERA_KEYS
]
DIT_POSITION_NAMES = ["state_tokens_mean", "action_tokens_mean"]


class CaptureError(RuntimeError):
    """A capture invariant (fire count, span stability, shape) failed."""


class ActivationCapture:
    """Forward hooks over a loaded ``XR1Runner`` model + per-inference
    invariant checks. Single-threaded, like the runner's noise override."""

    def __init__(self, runner: XR1Runner):
        self.runner = runner
        model = runner.model
        self.video_token_id = int(model.config.vlm_config.video_token_id)

        vlm_layers = model.vlm.model.language_model.layers
        dit_layers = model.dit.layers
        if len(vlm_layers) != VLM_LAYERS:
            raise CaptureError(f"expected {VLM_LAYERS} VLM layers, got {len(vlm_layers)}")
        if len(dit_layers) != DIT_LAYERS:
            raise CaptureError(f"expected {DIT_LAYERS} DiT layers, got {len(dit_layers)}")

        self._active = False
        self._reset_buffers()
        self.spans: list[list[tuple[int, int]]] | None = None  # first-seen, global

        model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        for i, layer in enumerate(vlm_layers):
            layer.register_forward_hook(self._make_vlm_hook(i))
        for j, layer in enumerate(dit_layers):
            layer.register_forward_hook(self._make_dit_hook(j))
        model.state_projector.register_forward_hook(self._state_embed_hook)

    # -- per-inference state ------------------------------------------------

    def _reset_buffers(self) -> None:
        self._pre_hook_fires = 0
        self._vlm_fires = [0] * VLM_LAYERS
        self._dit_fires = [0] * DIT_LAYERS
        self._state_fires = 0
        self._request_spans: list[list[tuple[int, int]]] | None = None
        self._seq_len: int | None = None
        self._vlm_out = np.zeros((VLM_LAYERS, len(VLM_POSITION_NAMES), VLM_HIDDEN), np.float16)
        self._dit_out = np.zeros(
            (DIT_LAYERS, len(DIT_FLOW_STEPS), len(DIT_POSITION_NAMES), DIT_HIDDEN),
            np.float16,
        )
        self._state_embed_out = np.zeros((STATE_TOKENS, DIT_HIDDEN), np.float16)

    # -- hooks --------------------------------------------------------------

    def _model_pre_hook(self, module, args, kwargs):
        if not self._active:
            return
        self._pre_hook_fires += 1
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise CaptureError("model called without input_ids kwarg")
        ids = input_ids.detach().cpu().numpy().reshape(-1)
        spans = video_token_spans(ids, self.video_token_id, num_cameras=len(CAMERA_KEYS))
        if self.spans is None:
            self.spans = spans
        elif spans != self.spans:
            raise CaptureError(
                f"video-token spans changed between requests: {spans} != {self.spans}"
            )
        self._request_spans = spans
        self._seq_len = int(ids.shape[0])

    def _make_vlm_hook(self, i: int):
        def hook(module, args, output):
            if not self._active:
                return
            self._vlm_fires[i] += 1
            if self._vlm_fires[i] > 1:
                raise CaptureError(f"VLM layer {i} fired more than once per inference")
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != VLM_HIDDEN:
                raise CaptureError(f"VLM layer {i}: unexpected output shape {tuple(hidden.shape)}")
            seq = hidden[0]
            if self._request_spans is None:
                raise CaptureError("VLM layer fired before the input_ids pre-hook")
            self._vlm_out[i, 0] = seq[-1].float().cpu().numpy().astype(np.float16)
            for ci, cam_spans in enumerate(self._request_spans):
                block = torch.cat([seq[s:e] for (s, e) in cam_spans], dim=0)
                self._vlm_out[i, 1 + ci] = (
                    block.float().mean(dim=0).cpu().numpy().astype(np.float16)
                )

        return hook

    def _make_dit_hook(self, j: int):
        def hook(module, args, output):
            if not self._active:
                return
            flow_step = self._dit_fires[j]  # n-th firing == Euler step n
            self._dit_fires[j] += 1
            if self._dit_fires[j] > DIT_NUM_STEPS:
                raise CaptureError(f"DiT layer {j} fired more than {DIT_NUM_STEPS} times")
            if flow_step not in DIT_FLOW_STEPS:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3:
                hidden = hidden[0]
            if hidden.shape != (DIT_SEQ_LEN, DIT_HIDDEN):
                raise CaptureError(f"DiT layer {j}: unexpected shape {tuple(hidden.shape)}")
            fi = DIT_FLOW_STEPS.index(flow_step)
            self._dit_out[j, fi, 0] = (
                hidden[DIT_STATE_SLICE].float().mean(dim=0).cpu().numpy().astype(np.float16)
            )
            self._dit_out[j, fi, 1] = (
                hidden[DIT_ACTION_SLICE].float().mean(dim=0).cpu().numpy().astype(np.float16)
            )

        return hook

    def _state_embed_hook(self, module, args, output):
        if not self._active:
            return
        self._state_fires += 1
        if self._state_fires > 1:
            raise CaptureError("state_projector fired more than once per inference")
        embed = output[0] if output.dim() == 3 else output
        if embed.shape != (STATE_TOKENS, DIT_HIDDEN):
            raise CaptureError(f"state_projector: unexpected shape {tuple(embed.shape)}")
        self._state_embed_out = embed.float().cpu().numpy().astype(np.float16)

    # -- capture entry point ------------------------------------------------

    def infer_and_capture(
        self,
        frames_by_cam: dict[str, np.ndarray],
        proprio_history: np.ndarray,
        instruction: str,
        noise_seed: int,
    ) -> dict[str, Any]:
        self._reset_buffers()
        noise = torch.from_numpy(noise_array(int(noise_seed)))
        self._active = True
        try:
            actions = self.runner.infer(
                frames_by_cam, proprio_history, instruction, noise=noise
            )
        finally:
            self._active = False

        # Fire-count asserts (the T4-contract pattern).
        if self._pre_hook_fires != 1:
            raise CaptureError(f"model pre-hook fired {self._pre_hook_fires}x, expected 1")
        bad_vlm = [i for i, n in enumerate(self._vlm_fires) if n != 1]
        if bad_vlm:
            raise CaptureError(f"VLM layers {bad_vlm} did not fire exactly once")
        bad_dit = [j for j, n in enumerate(self._dit_fires) if n != DIT_NUM_STEPS]
        if bad_dit:
            raise CaptureError(f"DiT layers {bad_dit} did not fire exactly {DIT_NUM_STEPS}x")
        if self._state_fires != 1:
            raise CaptureError(f"state_projector fired {self._state_fires}x, expected 1")
        if actions.shape != ACTIONS_SHAPE:
            raise CaptureError(f"actions shape {actions.shape} != {ACTIONS_SHAPE}")

        return {
            "vlm": self._vlm_out,
            "dit": self._dit_out,
            "state_embed": self._state_embed_out,
            "actions": actions.astype(np.float32),
            "seq_len": self._seq_len,
        }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def build_meta(capture: ActivationCapture, checkpoint_dir: str) -> dict[str, Any]:
    import transformers

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {
        "task": "plan-2 task 3: XR1 activation capture at replan steps",
        "hook_sites": {
            "vlm": "model.vlm.model.language_model.layers[i], i=0..35 (forward output, prefix pass fires once, KV cached)",
            "dit": "model.dit.layers[j], j=0..35 (forward output, fires 5x per inference; the n-th firing is Euler flow step n)",
            "state_embed": "model.state_projector (forward output, fires once)",
        },
        "arrays": {
            "vlm": f"(S, {VLM_LAYERS}, {len(VLM_POSITION_NAMES)}, {VLM_HIDDEN}) float16",
            "dit": f"(S, {DIT_LAYERS}, {len(DIT_FLOW_STEPS)}, {len(DIT_POSITION_NAMES)}, {DIT_HIDDEN}) float16",
            "state_embed": f"(S, {STATE_TOKENS}, {DIT_HIDDEN}) float16",
            "actions": f"(S, {ACTIONS_SHAPE[0]}, {ACTIONS_SHAPE[1]}) float32 (decoded, post decode_action)",
            "steps": "(S,) int64 recorded-episode step index of each capture (replan grid t=0,16,32,...)",
            "noise_seeds": "(S,) int64",
            "seq_len": "(S,) int64 VLM prefix length per capture (instruction-dependent)",
        },
        "vlm_positions": VLM_POSITION_NAMES,
        "dit_positions": DIT_POSITION_NAMES,
        "dit_flow_steps": list(DIT_FLOW_STEPS),
        "dit_token_layout": "[sink(1), state(4), noisy_action(16)] -> seq len 21",
        "state_token_note": (
            "The capture contract's 'state_token' position: this checkpoint's VLM "
            "prefix carries NO state token (state enters the DiT sequence via "
            "model.state_projector, modeling_mibot.py:1850), so it is served by the "
            "state_embed array plus each DiT layer's state_tokens_mean position; "
            "the VLM position set is last_prefix_token + 3 per-camera image means."
        ),
        "video_token_id": capture.video_token_id,
        "image_token_spans_per_camera": {
            key: [list(span) for span in spans]
            for key, spans in zip(CAMERA_KEYS, capture.spans or [])
        },
        "token_span_note": (
            "Half-open [start, end) index runs of video tokens in the tokenized "
            "prompt, grouped per camera in prompt order (left, right, wrist). The "
            "three videos precede the instruction text, so these spans are constant "
            "across episodes and steps; asserted on every request. last_prefix_token "
            "is index seq_len-1 (varies with instruction length, see seq_len)."
        ),
        "noise": (
            "np.random.default_rng(noise_seed).standard_normal((1,16,60)).astype(f32), "
            "noise_seed = episode_seed*100000 + step; injected via "
            "xr1_runner.override_initial_noise (strict single-draw scoped randn_like patch)"
        ),
        "fire_count_asserts": (
            "per inference: each VLM layer exactly 1, each DiT layer exactly 5, "
            "state_projector exactly 1, input_ids pre-hook exactly 1"
        ),
        "determinism": (
            "gate: one (episode, step-set) captured twice across two fresh server "
            "processes must be bit-identical (acts + actions + this meta); see "
            "output/mass_variation/activations/determinism_gate.json"
        ),
        "checkpoint_dir": str(checkpoint_dir),
        "git_sha": git_sha,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "env": {"MIBOT_SERVER_SEED": os.environ.get("MIBOT_SERVER_SEED")},
        "capture_dtype": "float16 (means computed in float32 from bf16 hidden states, then cast)",
    }


class CaptureServer:
    def __init__(self, port: int, acts_root: Path, checkpoint_dir: str, host: str = DEFAULT_HOST):
        self.host = host
        self.port = port
        self.acts_root = Path(acts_root)
        self.checkpoint_dir = str(checkpoint_dir)
        print(f"Loading XR1Runner from {checkpoint_dir} ...", flush=True)
        self.capture = ActivationCapture(XR1Runner(checkpoint_dir=checkpoint_dir))
        print("Model loaded.", flush=True)
        self.buffers: dict[str, dict[str, list]] = {}
        self.meta_written = False

    # -- per-message handlers ----------------------------------------------

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        msg_type = msg.get("type")
        if msg_type == "ping":
            return {"ok": True, "type": "pong", "acts_root": str(self.acts_root)}
        if msg_type == "infer":
            return self._handle_infer(msg)
        if msg_type == "finalize":
            return self._handle_finalize(msg)
        if msg_type == "discard":
            self.buffers.pop(msg["episode_id"], None)
            return {"ok": True}
        raise ValueError(f"unknown message type {msg_type!r}")

    def _handle_infer(self, msg: dict[str, Any]) -> dict[str, Any]:
        record = self.capture.infer_and_capture(
            msg["frames_by_cam"], msg["proprio_history"], msg["instruction"],
            msg["noise_seed"],
        )
        buf = self.buffers.setdefault(
            msg["episode_id"],
            {"steps": [], "noise_seeds": [], "seq_len": [], "vlm": [], "dit": [],
             "state_embed": [], "actions": []},
        )
        buf["steps"].append(int(msg["step"]))
        buf["noise_seeds"].append(int(msg["noise_seed"]))
        buf["seq_len"].append(int(record["seq_len"]))
        buf["vlm"].append(record["vlm"])
        buf["dit"].append(record["dit"])
        buf["state_embed"].append(record["state_embed"])
        buf["actions"].append(record["actions"])

        if not self.meta_written:
            self._write_meta()
        return {"ok": True, "actions": record["actions"], "step": int(msg["step"])}

    def _handle_finalize(self, msg: dict[str, Any]) -> dict[str, Any]:
        episode_id = msg["episode_id"]
        buf = self.buffers.pop(episode_id, None)
        if buf is None:
            raise ValueError(f"finalize for unknown episode {episode_id!r}")
        expected = list(msg["expected_steps"])
        if buf["steps"] != expected:
            raise ValueError(
                f"{episode_id}: captured steps {buf['steps']} != expected {expected}"
            )
        arrays = {
            "steps": np.asarray(buf["steps"], dtype=np.int64),
            "noise_seeds": np.asarray(buf["noise_seeds"], dtype=np.int64),
            "seq_len": np.asarray(buf["seq_len"], dtype=np.int64),
            "vlm": np.stack(buf["vlm"]),
            "dit": np.stack(buf["dit"]),
            "state_embed": np.stack(buf["state_embed"]),
            "actions": np.stack(buf["actions"]),
        }
        for key, value in msg.get("scalars", {}).items():
            arrays[key] = np.asarray(value)

        acts_path = self.acts_root / f"{episode_id}.npz"
        acts_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = acts_path.with_suffix(".npz.tmp.npz")
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, acts_path)
        acts_bytes = acts_path.stat().st_size
        print(
            f"[capture] wrote {acts_path} ({len(expected)} steps, {acts_bytes} bytes)",
            flush=True,
        )
        return {"ok": True, "acts_path": str(acts_path), "acts_bytes": acts_bytes}

    def _write_meta(self) -> None:
        meta = build_meta(self.capture, self.checkpoint_dir)
        meta_path = self.acts_root / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if existing != meta:
                raise CaptureError(
                    f"{meta_path} exists with different content than this server "
                    "would write (checkpoint/prompt/env changed?) -- refusing to "
                    "mix captures"
                )
        else:
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"[capture] wrote {meta_path}", flush=True)
        self.meta_written = True

    # -- socket loop (deploy/server.py shape) -------------------------------

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"Capture server running on {self.host}:{self.port} ...", flush=True)
            while True:
                conn, _addr = server_socket.accept()
                try:
                    while True:
                        msg = recv_message(conn)
                        if msg is None:
                            break
                        try:
                            response = self.handle(msg)
                        except Exception as exc:  # noqa: BLE001 -- reported to client
                            traceback.print_exc()
                            response = {
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            }
                        conn.sendall(pack_message(response))
                except Exception as exc:  # noqa: BLE001
                    print(f"Connection error: {exc}", flush=True)
                    traceback.print_exc()
                finally:
                    conn.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XR1 activation capture server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--acts-root", required=True)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server = CaptureServer(
        port=args.port, acts_root=Path(args.acts_root),
        checkpoint_dir=args.checkpoint_dir, host=args.host,
    )
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

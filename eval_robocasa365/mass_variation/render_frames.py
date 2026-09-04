# Copyright (C) 2026 Xiaomi Corporation.
"""Plan-2 amendment C: camera-frame re-render from the bit-exact replays.

The vision certificate (amendment C) asks whether each model's CAMERA
channel, in that model's own visual format, carries the hidden mass. The
Phase-1 npz never stored frames (only state/force/event channels), so the
frames have to come back from the SAME bit-exact replay scaffold Task 1's
``extract_policy_state.py`` uses -- no new policy rollouts, no new physics:

- ``render_episode.parse_npz_path`` -- (cell, condition, seed) from the path.
- ``render_episode.setup_condition_physics`` -- re-derives the density
  override from the INTENDED condition (``conditions.condition_physics``),
  never from the npz's measured ``mass_kg`` (that would make the replay
  circular -- see ``render_episode.py``'s module docstring). The measured
  mass/CoM are a post-setup assert at ``atol=1e-9``.
- The same strict success / liftoff_step / grasped / obj_pos bit-exactness
  gate: a divergence raises ``ReplayDivergence``, writes NOTHING for that
  episode, and is recorded in the manifest. Never loosened.

What is new here: the replay reads the three camera keys off each post-step
observation (``entry.CAMERA_KEYS``, rendered by robocasa's gym wrapper at
its default 256x256 -- exactly the frames the live Phase-1 policy loop saw)
and stores a downsampled 96x96 GRAYSCALE copy of the steps the certificate
needs.

Which steps (``frame_steps_needed``, pure + unit-tested)
--------------------------------------------------------
Physics must still step through the WHOLE episode (that is what makes the
replay bit-exact), but only the certificate's own rows need storing:

- every ``precontact`` step (amendment C Section 3's control mask), and
- for every ``carry`` step ``t``, the four steps XR1's visual format reads
  at ``t`` -- ``t, t-2, t-4, t-6``, left-clamped at 0 (``entry.
  sample_history(k=4, interval=2)``'s index rule).

That union is a superset of pi0.5's needs (its format is the single frame at
``t``), so ONE stored frame set serves both models' formats. It keeps ~69%
(xr1) / ~45% (pi0.5) of steps instead of 100%.

Storage is READER-side, not model-side
--------------------------------------
The stored 96x96 grayscale is NOT any model's input format -- both models
see 256x256 RGB. Downsampling to 96x96 grayscale (and any further reduction
the readers apply) is part of the READER, and is disclosed as such in the
certificate JSON's ``input_channels`` and in the report. What the MODEL's
format determines here is only WHICH frames and HOW MANY cameras each
reader may look at (amendment C Section 1).

Grayscale/resize are done with PIL, deterministically: ``Image.convert("L")``
(ITU-R 601-2 luma) then ``resize((96, 96), BILINEAR)``.

Run under the robocasa venv with EGL, same as ``extract_policy_state.py``:

    MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.render_frames [--limit N] \\
        [--time-budget-s S]

Resumable (an episode whose frames npz exists, or whose key is already in
the manifest's ``divergences``, is skipped), so bounded FOREGROUND
invocations can be chained until the corpus is complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval_robocasa365.entry import CAMERA_KEYS
from eval_robocasa365.mass_variation.analysis.dataset import phase_masks
from eval_robocasa365.mass_variation.conditions import episode_seeds
from eval_robocasa365.mass_variation.entry_mass import PRIMARY_CATEGORY, PRIMARY_ENV
from eval_robocasa365.mass_variation.extract_policy_state import (
    MASS_CONDITIONS,
    MODELS,
    cell_dir_for,
    load_manifest,
    phase1_npz_path,
    save_manifest,
)
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

log = logging.getLogger("render_frames")

# Storage side (READER-side reduction -- see the module docstring).
STORE_SIDE_PX = 96
# The resolution robocasa's gym wrapper renders at by default, i.e. what the
# live Phase-1 loop actually fed the policies (PandaOmronKeyConverter.
# get_camera_config -> 256x256). Asserted at replay time, never assumed.
SOURCE_SIDE_PX = 256
# XR1's visual format (entry.py --obs-history / --obs-interval defaults),
# which drives the frame keep-set; pi0.5's single frame is the k=1 subset.
K_FRAMES = 4
STRIDE_FRAMES = 2


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def frame_steps_needed(
    masks: dict[str, np.ndarray],
    k: int = K_FRAMES,
    stride: int = STRIDE_FRAMES,
) -> np.ndarray:
    """Sorted, unique step indices this episode must store for the vision
    certificate: every ``precontact`` step, plus each ``carry`` step's whole
    ``k``-frame stride-``stride`` left-clamped history window.

    ``masks`` is a ``dataset.phase_masks`` result (only ``precontact`` and
    ``carry`` are read). Returns an ``int64`` array; empty if both masks are
    empty.
    """
    if k < 1 or stride < 1:
        raise ValueError(f"frame_steps_needed: k and stride must be >= 1, got {k}/{stride}")
    precontact = np.asarray(masks["precontact"], dtype=bool)
    carry = np.asarray(masks["carry"], dtype=bool)
    if precontact.shape != carry.shape:
        raise ValueError(
            f"frame_steps_needed: mask shapes differ ({precontact.shape} vs {carry.shape})"
        )
    needed = np.flatnonzero(precontact).tolist()
    carry_steps = np.flatnonzero(carry)
    for offset in range(k):
        needed.extend(np.maximum(0, carry_steps - offset * stride).tolist())
    return np.unique(np.asarray(needed, dtype=np.int64))


def to_grayscale_thumb(frame: Any, side_px: int = STORE_SIDE_PX) -> np.ndarray:
    """One rendered RGB camera frame ``(H, W, 3) uint8`` -> ``(side, side)``
    uint8 grayscale, via PIL ``convert("L")`` (ITU-R 601-2 luma) then a
    BILINEAR resize. Deterministic; the ONLY lossy step between the replay
    and the certificate's inputs, and disclosed as reader-side."""
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"to_grayscale_thumb: expected (H, W, 3) uint8, got {array.shape}")
    image = Image.fromarray(array).convert("L")
    if image.size != (side_px, side_px):
        image = image.resize((side_px, side_px), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def frames_npz_path(frames_root: Path, model: str, condition: str, seed: int) -> Path:
    return Path(frames_root) / model / condition / f"ep_{seed}.npz"


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------


def render_episode_frames(
    npz_path: str | Path,
    out_path: str | Path,
    obj_name: str = DEFAULT_OBJ_NAME,
    category: str = PRIMARY_CATEGORY,
    store_side_px: int = STORE_SIDE_PX,
    source_side_px: int = SOURCE_SIDE_PX,
    k: int = K_FRAMES,
    stride: int = STRIDE_FRAMES,
) -> dict[str, Any]:
    """Bit-exact replay ``npz_path`` and write ``out_path`` with
    ``frames (T_kept, 3, side, side) uint8`` (camera axis in
    ``entry.CAMERA_KEYS`` order) and ``steps (T_kept,) int64``.

    Raises ``ReplayDivergence`` (writing nothing) on any
    success/liftoff_step/grasped/obj_pos mismatch against the recording --
    the same gate ``render_episode.py`` and ``extract_policy_state.py``
    enforce, unchanged.
    """
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import convert_action

    npz_path = Path(npz_path)
    out_path = Path(out_path)

    with np.load(npz_path) as data:
        npz = {key: data[key] for key in data.files}

    cell, condition, seed_from_path = parse_npz_path(npz_path)
    seed = int(npz["seed"])
    if seed != seed_from_path:
        raise ValueError(
            f"{npz_path}: seed mismatch -- path implies {seed_from_path}, "
            f"npz 'seed' scalar says {seed}"
        )

    recorded_actions = np.asarray(npz["actions"], dtype=np.float32)
    steps = recorded_actions.shape[0]
    recorded_success = bool(npz["success"])
    recorded_liftoff = int(npz["liftoff_step"])
    recorded_grasped = np.asarray(npz["grasped"], dtype=bool)
    recorded_obj_pos = np.asarray(npz["obj_pos"], dtype=np.float32)
    recorded_mass_kg = float(npz["mass_kg"])
    recorded_com_offset_m = float(npz["com_offset_m"])

    # Keep-set from the RECORDED events (not the replay's), so it is known
    # before the replay starts and is provably the certificate's own rows.
    masks = phase_masks(
        recorded_grasped, recorded_liftoff, recorded_obj_pos[:, 2],
        float(recorded_obj_pos[0, 2]),
    )
    keep_steps = frame_steps_needed(masks, k=k, stride=stride)
    keep_set = set(int(s) for s in keep_steps)

    log.info(
        "Rendering %s: condition=%s seed=%d steps=%d keep=%d (%.0f%%)",
        npz_path, condition, seed, steps, len(keep_steps),
        100.0 * len(keep_steps) / max(1, steps),
    )

    env = gym.make(f"robocasa/{cell}", split="pretrain", obj_groups=category, seed=seed)
    kept: list[np.ndarray] = []
    try:
        setup = setup_condition_physics(env, condition, seed, obj_name=obj_name)
        _assert_close("mass_kg", setup["mass_kg"], recorded_mass_kg, MASS_ATOL_KG, npz_path)
        _assert_close(
            "com_offset_m", setup["com_offset_m"], recorded_com_offset_m,
            COM_OFFSET_ATOL_M, npz_path,
        )

        recorder = StepRecorder()
        info: dict[str, Any] = {}
        for t in range(steps):
            observation, _, done, truncated, info = env.step(
                convert_action(recorded_actions[t])
            )
            recorder.record(env, obj_name, recorded_actions[t])
            if t in keep_set:
                per_camera = []
                for key in CAMERA_KEYS:
                    frame = np.asarray(observation[key], dtype=np.uint8)
                    if frame.shape[:2] != (source_side_px, source_side_px):
                        raise ValueError(
                            f"{npz_path}: camera {key} rendered at {frame.shape[:2]}, "
                            f"expected ({source_side_px}, {source_side_px}) -- the "
                            "Phase-1 loop's own resolution"
                        )
                    per_camera.append(to_grayscale_thumb(frame, store_side_px))
                kept.append(np.stack(per_camera, axis=0))

        replayed_success = bool(info.get("success", False))
    finally:
        env.close()

    # Same bit-exactness gate as extract_policy_state.py: finalize the
    # replay's own recorder trace to a throwaway npz, compare, discard.
    replay_npz_path = out_path.with_suffix(".replay_check.npz")
    replay_npz_path.parent.mkdir(parents=True, exist_ok=True)
    recorder.finalize(
        replay_npz_path,
        mass_kg=setup["mass_kg"], com_offset_m=setup["com_offset_m"],
        com_axis=setup["com_axis"], seed=seed, success=replayed_success,
    )
    try:
        with np.load(replay_npz_path) as replay_data:
            replayed_liftoff = int(replay_data["liftoff_step"])
            replayed_grasped = np.asarray(replay_data["grasped"], dtype=bool)
            replayed_obj_pos = np.asarray(replay_data["obj_pos"], dtype=np.float32)
    finally:
        replay_npz_path.unlink(missing_ok=True)

    divergences = []
    if replayed_success != recorded_success:
        divergences.append(f"success: replayed={replayed_success} recorded={recorded_success}")
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
        divergences.append(f"obj_pos trace differs from the recording (max |diff|={max_diff:.3e} m)")

    if divergences:
        raise ReplayDivergence(f"{npz_path}: replay diverged: " + "; ".join(divergences))

    frames = (
        np.stack(kept, axis=0).astype(np.uint8)
        if kept
        else np.zeros((0, len(CAMERA_KEYS), store_side_px, store_side_px), dtype=np.uint8)
    )
    expected = (len(keep_steps), len(CAMERA_KEYS), store_side_px, store_side_px)
    if frames.shape != expected:
        raise ValueError(f"{npz_path}: frames shape {frames.shape} != expected {expected}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frames=frames,
        steps=keep_steps,
        seed=np.int64(seed),
        episode_steps=np.int64(steps),
        camera_keys=np.array(list(CAMERA_KEYS)),
        store_side_px=np.int64(store_side_px),
        source_side_px=np.int64(source_side_px),
        keep_k=np.int64(k),
        keep_stride=np.int64(stride),
    )
    log.info("Wrote %s (%d/%d steps kept, bit-exact replay verified)",
             out_path, len(keep_steps), steps)

    return {
        "npz_path": str(npz_path),
        "out_path": str(out_path),
        "cell": cell,
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "kept": int(len(keep_steps)),
        "success": replayed_success,
        "liftoff_step": replayed_liftoff,
        "bit_exact": True,
    }


# ---------------------------------------------------------------------------
# CLI: batch over both corpora, resumable, manifest
# ---------------------------------------------------------------------------


def build_pending(
    phase1_root: Path,
    frames_root: Path,
    cell: str,
    models: list[str],
    conditions: list[str],
    seeds: list[int],
    manifest: dict[str, Any],
) -> list[tuple[str, str, int, Path, Path]]:
    """Every ``(model, condition, seed)`` whose Phase-1 npz exists but whose
    frames npz does not, excluding keys already recorded as a divergence
    (same rule as ``extract_policy_state.build_pending``)."""
    pending = []
    for model in models:
        for condition in conditions:
            for seed in seeds:
                key = f"{model}/{condition}/ep_{seed}"
                if key in manifest.get("divergences", {}):
                    continue
                p1_path = phase1_npz_path(phase1_root, cell, model, condition, seed)
                if not p1_path.exists():
                    continue
                out_path = frames_npz_path(frames_root, model, condition, seed)
                if out_path.exists():
                    continue
                pending.append((model, condition, seed, p1_path, out_path))
    return pending


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan-2 amendment C: re-render camera frames from the bit-exact replays."
    )
    parser.add_argument("--phase1-root", default="output/mass_variation/phase1")
    parser.add_argument("--frames-root", default="output/mass_variation/frames")
    parser.add_argument("--cell", default=PRIMARY_ENV)
    parser.add_argument("--category", default=PRIMARY_CATEGORY)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--conditions", nargs="+", default=list(MASS_CONDITIONS))
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=35)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many pending episodes this invocation.")
    parser.add_argument("--time-budget-s", type=float, default=None,
                        help="Exit cleanly (resumable) once this much wall time has elapsed.")
    parser.add_argument("--store-side-px", type=int, default=STORE_SIDE_PX)
    parser.add_argument("--source-side-px", type=int, default=SOURCE_SIDE_PX)
    parser.add_argument("--manifest", default=None,
                        help="Default: <frames-root>/frames_manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    phase1_root = Path(args.phase1_root)
    frames_root = Path(args.frames_root)
    manifest_path = (
        Path(args.manifest) if args.manifest else frames_root / "frames_manifest.json"
    )
    seeds = episode_seeds(args.base_seed, args.cell_index, args.n_seeds)

    manifest = load_manifest(manifest_path)
    manifest.setdefault("config", {}).update({
        "store_side_px": args.store_side_px,
        "source_side_px": args.source_side_px,
        "keep_k": K_FRAMES,
        "keep_stride": STRIDE_FRAMES,
        "camera_keys": list(CAMERA_KEYS),
        "grayscale": "PIL convert('L') (ITU-R 601-2 luma) then BILINEAR resize",
        "note": ("96x96 grayscale storage is READER-side, not any model's "
                 "input format (both see 256x256 RGB) -- amendment C Section 1"),
    })
    pending = build_pending(
        phase1_root, frames_root, args.cell, args.models, args.conditions, seeds, manifest,
    )
    log.info("pending=%d (before --limit)", len(pending))
    if args.limit is not None:
        pending = pending[: args.limit]

    t0 = time.monotonic()
    n_ok = n_div = 0
    for model, condition, seed, p1_path, out_path in pending:
        if args.time_budget_s is not None and (time.monotonic() - t0) >= args.time_budget_s:
            log.info("BUDGET_EXIT: time budget reached; re-invoke to resume.")
            break
        key = f"{model}/{condition}/ep_{seed}"
        ep_t0 = time.monotonic()
        try:
            result = render_episode_frames(
                p1_path, out_path, category=args.category,
                store_side_px=args.store_side_px, source_side_px=args.source_side_px,
            )
            result["model"] = model
            result["wall_s"] = round(time.monotonic() - ep_t0, 1)
            manifest["episodes"][key] = result
            n_ok += 1
            log.info("OK %s: steps=%d kept=%d wall=%.1fs",
                     key, result["steps"], result["kept"], result["wall_s"])
        except ReplayDivergence as exc:
            log.error("DIVERGED %s: %s", key, exc)
            manifest["divergences"][key] = {
                "model": model, "condition": condition, "seed": seed,
                "npz_path": str(p1_path), "error": str(exc),
            }
            n_div += 1
        save_manifest(manifest_path, manifest)

    remaining = build_pending(
        phase1_root, frames_root, args.cell, args.models, args.conditions, seeds, manifest,
    )
    summary = {
        "status": "complete" if not remaining else "incomplete",
        "rendered_this_run": n_ok,
        "diverged_this_run": n_div,
        "remaining": len(remaining),
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

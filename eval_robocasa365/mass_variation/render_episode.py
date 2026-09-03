# Copyright (C) 2026 Xiaomi Corporation.
"""npz -> mp4 bit-exact replay renderer (Plan 1 final-review fix wave,
user-requested item 2).

Reads one recorded Phase-1 episode npz (``recorder.StepRecorder.finalize``
output, at ``.../phase1/<cell_dir>/<condition>/ep_<seed>.npz``) and replays
it against a FRESH env instance to produce an mp4:

1. Reseed a fresh ``gym.make(...)`` env with the npz's own ``seed`` (parsed
   from BOTH the filename and the ``seed`` scalar; they must agree).
2. Re-derive the density override EXACTLY as ``entry_mass.
   run_condition_episode`` did for the original run -- from the INTENDED
   condition's mass (``conditions.condition_physics``, condition parsed
   from the npz's own directory name, e.g. ``.../MassMedium/ep_37.npz`` ->
   ``"MassMedium"``) composed with a freshly-probed default-density mass
   (``overrides.mass_to_density``), exactly the same two-call composition
   ``run_condition_episode`` performs. This is deliberately NOT derived
   from the npz's measured ``mass_kg`` scalar -- doing that would make the
   replay circular (it would only prove ``overrides.py`` can reproduce a
   number it was handed, not that the recorded ``conditions.py`` +
   ``overrides.py`` composition an actual Phase-1 run performs is itself
   reproducible). The measured ``mass_kg``/``com_offset_m`` are used ONLY
   as a post-setup verification assert, at ``atol=1e-9`` -- tight, because
   this is a determinism replay (same seed must reproduce the identical
   live sim state), not a fresh episode where a percent tolerance would be
   appropriate (contrast ``run_condition_episode``'s ``mass_tol_pct``).
3. Step every recorded action (``npz["actions"]``, the RAW pre-
   ``convert_action`` actions) through the SAME ``StepRecorder`` class the
   original run used, so the replay's own grasped/obj_pos/liftoff_step/
   success trace can be asserted against the original npz's -- not just
   the two headline scalars (success, liftoff_step) the fix-wave spec
   names, but the full per-step arrays those scalars are derived from.
4. Render the over-the-shoulder third-person camera
   (``video.robot0_agentview_left`` -- robocasa's ``camera_utils.py``
   positions the ``agentview_left``/``agentview_right`` pair as the
   robot-relative third-person views; there is no camera literally named
   "over_shoulder" in robocasa, so this is the closest match, chosen over
   the wrist camera and over ``entry.make_video_frame``'s 3-camera
   side-by-side montage per the fix-wave item's "over_shoulder camera"
   wording) at 20 fps (matching ``metrics.CONTROL_HZ`` and
   ``entry.py --video-fps``'s default) to an mp4 via ``imageio.mimsave``.

If the replay diverges from the recording (physics re-derivation off by
more than the assert tolerances, or the replayed success/liftoff/grasped/
obj_pos trace disagrees with the npz's), this raises ``ReplayDivergence``
and does NOT render an mp4 for that episode -- the acceptance bar is a
video that is PROVEN to reproduce the recorded episode, and loosening the
assert to force a render through would make the video misleading. Run
under the robocasa venv with EGL, same as ``run_phase1.py``:

    MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.render_episode <npz_path> [<npz_path> ...]

    MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.render_episode --batch <list_file>

``--batch`` takes a text file with one npz path per line (blank lines and
``#``-prefixed lines ignored) -- for rendering a fixed demo set without
retyping paths on the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from eval_robocasa365.mass_variation.conditions import condition_physics
from eval_robocasa365.mass_variation.entry_mass import PRIMARY_CATEGORY
from eval_robocasa365.mass_variation.overrides import (
    apply_com_offset,
    axis_index,
    install_density_override,
    mass_to_density,
    settle_and_gate,
    uninstall_density_override,
)
from eval_robocasa365.mass_variation.recorder import StepRecorder

log = logging.getLogger("render_episode")

# The "over-the-shoulder" third-person view -- see the module docstring's
# camera note. Rendering fps mirrors metrics.CONTROL_HZ / entry.py's
# --video-fps default (both 20).
CAMERA_KEY = "video.robot0_agentview_left"
FPS = 20
DEFAULT_OBJ_NAME = "obj"
DEFAULT_PROBE_DENSITY = 100.0
MASS_ATOL_KG = 1e-9
COM_OFFSET_ATOL_M = 1e-9


class ReplayDivergence(RuntimeError):
    """Raised when a replayed episode does not reproduce the npz it is
    replaying, within this script's tolerances. Never loosen these
    tolerances to force a render through -- per the fix-wave spec, a
    divergence must be reported honestly, not hidden."""


def parse_npz_path(npz_path: Path) -> tuple[str, str, int]:
    """(cell, condition, seed) from a Phase-1 npz path of the shape
    ``.../phase1/<cell_dir>/<condition>/ep_<seed>.npz``.

    ``cell_dir`` may carry a ``__<model>`` suffix (``run_phase1.
    cell_dir_for``'s T7 convention, e.g. ``PickPlaceCounterToCabinet__
    pi05_robocasa``); the true cell name is the part before it.
    """
    npz_path = Path(npz_path)
    condition = npz_path.parent.name
    cell_dir = npz_path.parent.parent.name
    cell = cell_dir.split("__", 1)[0]
    stem = npz_path.stem  # "ep_<seed>"
    if not stem.startswith("ep_"):
        raise ValueError(
            f"{npz_path}: expected an 'ep_<seed>.npz' filename, got {npz_path.name!r}"
        )
    seed = int(stem[len("ep_"):])
    return cell, condition, seed


def setup_condition_physics(
    env: Any,
    condition: str,
    seed: int,
    obj_name: str = DEFAULT_OBJ_NAME,
    probe_density: float = DEFAULT_PROBE_DENSITY,
) -> dict[str, Any]:
    """Reproduce EXACTLY the physics-injection sequence ``entry_mass.
    run_condition_episode`` performs for ``condition`` -- re-derived from
    ``conditions.condition_physics`` (the INTENDED condition, not a
    measured scalar trusted from a recording), so this is the identical
    composition an actual Phase-1 run performs, not a shortcut.

    Density override state is left uninstalled on return either way (this
    function's own try/finally): the override only matters at
    ``env.reset()`` construction time (``overrides.install_density_
    override`` patches ``MJCFObject.__init__``), never during stepping, so
    it is safe -- and necessary for ``--batch`` re-entry across multiple
    npz -- to uninstall it immediately after the reset that matters,
    rather than holding it installed for the whole episode the way
    ``run_condition_episode``'s single outer ``finally`` does (that
    difference is structural only; it changes nothing observable).

    Returns ``{"initial_observation", "mass_kg", "com_offset_m"}`` -- the
    observation from the reset that matters, plus the LIVE measured
    physics (read fresh off ``env.sim``, mirroring ``run_condition_
    episode``'s own assert-at-record pattern) for the caller to verify
    against the npz's recorded scalars.
    """
    physics = condition_physics(condition)
    is_com_condition = physics["com_offset_m"] != 0.0

    if is_com_condition:
        # Two-pass reset pattern (overrides.py module docstring / entry_mass
        # run_condition_episode's CoM branch): pass 1 discovers the mesh's
        # authored ipos + probe mass and gives this run's own center-CoM
        # settle reference; pass 2 installs density for this condition's
        # mass level and applies the CoM offset immediately after THAT
        # reset returns.
        env.reset(seed=seed)
        bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
        probe_mass_kg = float(env.sim.model.body_mass[bid])
        authored_ipos = np.array(env.sim.model.body_ipos[bid], dtype=float, copy=True)
        center_settle = settle_and_gate(env, obj_name)
        reference_pose = {
            "pos": center_settle["post_pos"],
            "quat": center_settle["post_quat"],
        }

        target_density = mass_to_density(
            physics["mass_kg"], probe_mass_kg, probe_density=probe_density
        )
        install_density_override({obj_name: target_density})
        try:
            initial_observation, _ = env.reset(seed=seed)
            bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
        finally:
            uninstall_density_override()

        apply_com_offset(env, obj_name, physics["com_offset_m"], physics["com_axis"])
        settle_and_gate(env, obj_name, reference_pose=reference_pose)

        axis_idx = axis_index(physics["com_axis"])
        actual_com_offset_m = float(
            env.sim.model.body_ipos[bid][axis_idx] - authored_ipos[axis_idx]
        )
    else:
        # Mass condition: one probe reset (density uninstalled) to measure
        # the default-density mass of whatever mesh this seed samples, then
        # the ONE reset that matters with the target density installed.
        env.reset(seed=seed)
        bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
        probe_mass_kg = float(env.sim.model.body_mass[bid])

        target_density = mass_to_density(
            physics["mass_kg"], probe_mass_kg, probe_density=probe_density
        )
        install_density_override({obj_name: target_density})
        try:
            initial_observation, _ = env.reset(seed=seed)
            bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
        finally:
            uninstall_density_override()
        actual_com_offset_m = 0.0  # never applied for a mass condition.

    actual_mass_kg = float(env.sim.model.body_mass[bid])
    return {
        "initial_observation": initial_observation,
        "mass_kg": actual_mass_kg,
        "com_offset_m": actual_com_offset_m,
        "com_axis": physics["com_axis"],
    }


def _assert_close(label: str, replayed: float, recorded: float, atol: float, npz_path: Path) -> None:
    diff = abs(replayed - recorded)
    if diff > atol:
        raise ReplayDivergence(
            f"{npz_path}: replayed {label} {replayed!r} diverges from the "
            f"recorded {recorded!r} by {diff:.3e} (atol {atol}) -- physics "
            "re-derivation is not bit-exact"
        )


def render_episode(
    npz_path: str | Path,
    out_path: str | Path,
    obj_name: str = DEFAULT_OBJ_NAME,
    category: str = PRIMARY_CATEGORY,
    fps: int = FPS,
    camera_key: str = CAMERA_KEY,
) -> dict[str, Any]:
    """Bit-exact replay ``npz_path`` and write ``out_path`` (mp4). Returns a
    result dict on success; raises ``ReplayDivergence`` on any mismatch
    (no partial/best-effort mp4 is written in that case)."""
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

    log.info(
        "Replaying %s: cell=%s condition=%s seed=%d steps=%d recorded_success=%s "
        "recorded_liftoff_step=%d",
        npz_path, cell, condition, seed, steps, recorded_success, recorded_liftoff,
    )

    env = gym.make(f"robocasa/{cell}", split="pretrain", obj_groups=category, seed=seed)
    frames: list[np.ndarray] = []
    try:
        setup = setup_condition_physics(env, condition, seed, obj_name=obj_name)

        _assert_close("mass_kg", setup["mass_kg"], recorded_mass_kg, MASS_ATOL_KG, npz_path)
        _assert_close(
            "com_offset_m", setup["com_offset_m"], recorded_com_offset_m,
            COM_OFFSET_ATOL_M, npz_path,
        )

        observation = setup["initial_observation"]
        frames.append(np.ascontiguousarray(observation[camera_key], dtype=np.uint8))

        recorder = StepRecorder()
        info: dict[str, Any] = {}
        for t in range(steps):
            observation, _, done, truncated, info = env.step(
                convert_action(recorded_actions[t])
            )
            recorder.record(env, obj_name, recorded_actions[t])
            frames.append(np.ascontiguousarray(observation[camera_key], dtype=np.uint8))

        replayed_success = bool(info.get("success", False))
    finally:
        env.close()

    replay_npz_path = out_path.with_suffix(".replay.npz")
    recorder.finalize(
        replay_npz_path,
        mass_kg=setup["mass_kg"],
        com_offset_m=setup["com_offset_m"],
        com_axis=setup["com_axis"],
        seed=seed,
        success=replayed_success,
    )
    with np.load(replay_npz_path) as replay_data:
        replayed_liftoff = int(replay_data["liftoff_step"])
        replayed_grasped = np.asarray(replay_data["grasped"], dtype=bool)
        replayed_obj_pos = np.asarray(replay_data["obj_pos"], dtype=np.float32)
    replay_npz_path.unlink()

    # The two scalars the fix-wave spec names explicitly, asserted exactly
    # (both are discrete: a bool and an int index).
    divergences = []
    if replayed_success != recorded_success:
        divergences.append(f"success: replayed={replayed_success} recorded={recorded_success}")
    if replayed_liftoff != recorded_liftoff:
        divergences.append(
            f"liftoff_step: replayed={replayed_liftoff} recorded={recorded_liftoff}"
        )
    # Extra per-step bit-exactness checks (stronger than the spec's bar,
    # not a substitute for it): the full grasped trace exactly, and obj_pos
    # to a tight float32-round-trip tolerance (0.1 mm) rather than exact
    # equality -- two independently-computed float32 stacks accumulated
    # over hundreds of MuJoCo steps.
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    log.info("Wrote %s (%d frames @ %d fps, bit-exact replay verified)", out_path, len(frames), fps)

    return {
        "npz_path": str(npz_path),
        "mp4_path": str(out_path),
        "cell": cell,
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "success": replayed_success,
        "liftoff_step": replayed_liftoff,
        "bit_exact": True,
    }


def out_path_for(out_dir: Path, npz_path: Path) -> Path:
    cell, condition, seed = parse_npz_path(npz_path)
    return out_dir / cell / condition / f"ep_{seed}.mp4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="npz -> mp4 bit-exact replay renderer.")
    parser.add_argument("npz", nargs="*", help="One or more npz paths.")
    parser.add_argument(
        "--batch", default=None,
        help="Text file with one npz path per line (blank/# lines ignored); "
        "overrides positional npz arguments.",
    )
    parser.add_argument("--out-dir", default="output/mass_variation/renders")
    parser.add_argument("--category", default=PRIMARY_CATEGORY)
    parser.add_argument("--obj-name", default=DEFAULT_OBJ_NAME)
    parser.add_argument("--camera-key", default=CAMERA_KEY)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args(argv)


def _npz_paths_for(args: argparse.Namespace) -> list[Path]:
    if args.batch:
        lines = Path(args.batch).read_text(encoding="utf-8").splitlines()
        paths = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        if not paths:
            raise SystemExit(f"--batch {args.batch}: no npz paths found")
        return [Path(p) for p in paths]
    if not args.npz:
        raise SystemExit("Pass one or more npz paths, or --batch <list file>.")
    return [Path(p) for p in args.npz]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    npz_paths = _npz_paths_for(args)
    out_dir = Path(args.out_dir)

    rendered: list[dict[str, Any]] = []
    diverged: list[dict[str, Any]] = []
    for npz_path in npz_paths:
        out_path = out_path_for(out_dir, npz_path)
        try:
            result = render_episode(
                npz_path, out_path,
                obj_name=args.obj_name, category=args.category,
                fps=args.fps, camera_key=args.camera_key,
            )
            rendered.append(result)
        except ReplayDivergence as exc:
            log.error("DIVERGED: %s", exc)
            diverged.append({"npz_path": str(npz_path), "error": str(exc)})

    summary = {"rendered": rendered, "diverged": diverged}
    print(json.dumps(summary, indent=2))
    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())

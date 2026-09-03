# Copyright (C) 2026 Xiaomi Corporation.
"""Study entry loop for the XR1 mass/CoM study (Plan 1, Task 3).

Composes Task 1 (``conditions.condition_physics`` / ``episode_seeds``) and
Task 2 (``overrides`` -- density/CoM injection, two-pass reset pattern) with
a rewritten episode loop and this task's ``recorder.StepRecorder`` into one
episode of one condition, saved to
``output/mass_variation/phase1/<cell>/<condition>/ep_<seed>.npz``.

``run_episode`` is the rewritten ~80-line loop the brief calls for -- the
stock loop in ``entry.evaluate_task`` has no per-step recorder hook, so it
is not reused wholesale; ``EvalClient``, ``reset_env``, and the pure
observation-queue helpers (``collect_images``, ``observation_to_state``,
``sample_history``) ARE imported from ``entry.py``, not copied, per the
brief's "Consumes" line. ``convert_action`` is imported directly from
``robocasa.utils.env_utils`` (where ``entry.py`` itself gets it, inside
``main()``, not as a module-level name there).

Controller ruling (T7 forward-compat): the model boundary is the
``PolicyClient`` protocol (``infer(obs_history, instruction) -> action
chunk`` + ``chunk_len``/``replan`` attributes). ``XR1SocketClient`` wraps
``entry.EvalClient`` behind it; Task 7 adds a pi0.5 client behind the same
protocol so ``run_episode``/the recorder/the seeds/the conditions are
unchanged across models.

Two-pass reset pattern (mirrors overrides.py's module docstring; repeated
here because ``run_condition_episode`` is what actually drives it): mass
conditions install the density override BEFORE the reset that matters --
density is baked in before ``Kitchen``'s internal settle loop runs, so one
reset is settle-consistent. CoM conditions must reset ONCE to discover the
sampled mesh/authored ipos, THEN reset AGAIN with the SAME seed (verified
deterministic resample) and apply the CoM offset immediately after that
second reset returns, per the settle-loop-runs-inside-reset finding in
Task 2. CoM conditions ALSO carry the "medium" mass level (Task 1's
``condition_physics``), so their second reset installs the density
override too -- both physics knobs land in the SAME reset.

Post-review fixes (binding)
------------------------------
Two bugs from the first pass, both about the CoM path silently not doing
what its own npz scalars claimed:

1. ``run_episode`` used to unconditionally call ``reset_env`` (a hard
   reset, which REBUILDS the MuJoCo model) as its very first action --
   which silently DISCARDED any CoM offset ``run_condition_episode`` had
   just applied, since that offset only lives in the model instance that
   reset just threw away. Fixed by giving ``run_episode`` an
   ``initial_observation`` parameter: when the caller already reset the
   env as part of physics setup (which every ``run_condition_episode``
   call does), it passes the observation THAT reset already returned, and
   ``run_episode`` performs NO reset of its own.
2. The CoM branch never installed the density override, so CoM episodes
   silently ran at whatever density the object's authored MJCF specifies
   (NOT the "medium" mass ``condition_physics`` says they carry).

Both are now covered by an assert-at-record check: right before
``run_episode`` starts (i.e. at step 0), ``run_condition_episode`` reads
the LIVE ``env.sim`` state (body_mass, and for CoM conditions the
body_ipos offset from the pass-1 authored baseline) and asserts it matches
the intended condition, then passes THOSE MEASURED values (not the
intended ones) as the npz's ``mass_kg``/``com_offset_m`` scalars -- so the
recorded ground truth is self-verifying: a regression of either bug above
raises immediately instead of silently writing a wrong-but-plausible npz.

Reset-count budget (IMPORTANT-3): mass conditions need only 1 reset in the
steady state and 2 on a cold start, thanks to a per-process
``_PROBE_MASS_CACHE`` keyed by ``(env_name, category, seed)`` -- a fixed
seed always samples the same mesh (deterministic reseed), so the mesh's
default-density probe mass only needs discovering ONCE per seed; every
other condition sharing that seed (there are 3 mass conditions per seed in
Phase 1, plus 2 deferred CoM conditions) installs its own target density
straight from the cached probe and pays only the 1 reset with the override
active. CoM conditions ALWAYS pay exactly 2 resets (pass 1's settle
REFERENCE pose is condition-specific and is never skipped, even on a
cache hit for the probe mass) but write-through into the same cache so a
mass condition for the same seed, run afterward, benefits too. At 105+
Phase 1 episodes this removes on the order of 70 redundant probe resets
(and their 20-step settle loops) versus probing fresh every episode.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from eval_robocasa365.entry import (
    CAMERA_KEYS,
    DEFAULT_MODEL_PATH,
    ROBOT_TYPE,
    EvalClient,
    collect_images,
    observation_to_state,
    reset_env,
    sample_history,
)
from eval_robocasa365.mass_variation.conditions import condition_physics, episode_seeds
from eval_robocasa365.mass_variation.overrides import (
    apply_com_offset,
    axis_index,
    install_density_override,
    mass_to_density,
    settle_and_gate,
    uninstall_density_override,
)
from eval_robocasa365.mass_variation.recorder import StepRecorder

PRIMARY_ENV = "PickPlaceCounterToCabinet"
PRIMARY_CATEGORY = "milk"
DEFAULT_OBJ_NAME = "obj"

# Per-process cache: (env_name, category, seed) -> the object's mass (kg) at
# the default probe density (100), i.e. what a reset with NO density
# override installed would measure. Populated by whichever condition for a
# given seed runs first (mass or CoM); read (not re-probed) by every later
# condition sharing that seed. See the module docstring's "Reset-count
# budget" note.
_PROBE_MASS_CACHE: dict[tuple[str, str, int], float] = {}


class PolicyClient(Protocol):
    """Forward-compat model boundary (T7 controller ruling). ``infer``
    takes the FULL observation history the loop maintains (state + a
    per-camera image stack, already downsampled to the policy's own
    history/stride) and returns an action chunk ``(chunk_len, act_dim)``.
    """

    chunk_len: int
    replan: int

    def infer(self, obs_history: dict[str, Any], instruction: str) -> np.ndarray: ...


class XR1SocketClient:
    """``PolicyClient`` wrapping ``entry.EvalClient`` over the stock pickle
    socket server (``deploy/server.py``). Task 7 adds a pi0.5 client behind
    the same protocol.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        host: str = "localhost",
        port: int = 10086,
        robot_type: str = ROBOT_TYPE,
        crop_ratio: float = 0.95,
        chunk_len: int = 16,
        replan: int = 16,
    ) -> None:
        self._client = EvalClient(
            model_path=model_path,
            host=host,
            port=port,
            robot_type=robot_type,
            crop_ratio=crop_ratio,
        )
        self.chunk_len = chunk_len
        self.replan = replan

    def infer(self, obs_history: dict[str, Any], instruction: str) -> np.ndarray:
        actions = self._client.infer(
            obs_history["state"], obs_history["images"], instruction
        )
        return np.asarray(actions, dtype=np.float32)

    def close(self) -> None:
        self._client.close()


def run_episode(
    env: Any,
    client: PolicyClient,
    condition_physics_dict: dict[str, Any],
    seed: int,
    horizon: int,
    obj_name: str = DEFAULT_OBJ_NAME,
    npz_path: str | Path | None = None,
    obs_history: int = 4,
    obs_interval: int = 2,
    grasp_fn: Any = None,
    initial_observation: dict[str, Any] | None = None,
) -> tuple[bool, int, Path | None]:
    """Roll out ONE episode on an already-constructed, already-physics-
    injected ``env`` (see the module docstring's two-pass reset note --
    physics injection happens BEFORE this call, via ``run_condition_episode``
    or a caller doing the same sequence), recording ground truth every step.

    ``condition_physics_dict`` (Task 1's ``condition_physics`` output, or --
    from ``run_condition_episode`` -- the MEASURED live-sim equivalent, see
    the module docstring's "Post-review fixes" note) is NOT applied here:
    it is folded into the saved npz's scalars (mass_kg, com_offset_m,
    com_axis) so the recorder output is self-describing without
    re-deriving the condition from the cell/seed at analysis time.

    ``initial_observation``: when the caller has ALREADY reset ``env`` as
    part of physics injection (the normal case, via
    ``run_condition_episode``), pass the observation THAT reset returned
    here -- this method then performs NO reset of its own (CRITICAL-1
    class of bug from the review: a reset here is a HARD reset that
    rebuilds the MuJoCo model, silently discarding any post-reset physics
    write, e.g. a CoM offset, the caller just applied). Only omit this
    (falling back to an internal ``reset_env(env, seed)`` call) for
    standalone/test use where ``env`` has NOT already been reset for this
    seed.

    Returns ``(success, steps, npz_path_written_or_None)``.
    """
    from robocasa.utils.env_utils import convert_action

    if initial_observation is not None:
        observation = initial_observation
    else:
        observation, _ = reset_env(env, seed)
    instruction = observation["annotation.human.task_description"]

    queue_length = (obs_history - 1) * obs_interval + 1
    image_queues = {key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS}
    state_queue: collections.deque[np.ndarray] = collections.deque(maxlen=queue_length)
    for key, image in collect_images(observation).items():
        image_queues[key].append(image)
    state_queue.append(observation_to_state(observation))

    recorder = StepRecorder()
    action_plan: collections.deque[np.ndarray] = collections.deque()

    success = False
    steps = 0
    while steps < horizon:
        if not action_plan:
            obs_hist = {
                "state": sample_history(state_queue, obs_history, obs_interval),
                "images": {
                    key: sample_history(queue, obs_history, obs_interval)
                    for key, queue in image_queues.items()
                },
            }
            action_chunk = np.asarray(client.infer(obs_hist, instruction))
            chunk_len = getattr(client, "chunk_len", len(action_chunk))
            if len(action_chunk) != chunk_len:
                raise RuntimeError(
                    f"Policy returned {len(action_chunk)} actions but its "
                    f"chunk_len is {chunk_len}"
                )
            replan = getattr(client, "replan", len(action_chunk))
            if len(action_chunk) < replan:
                raise RuntimeError(
                    f"Policy returned {len(action_chunk)} actions but "
                    f"replan is {replan}"
                )
            action_plan.extend(action_chunk[:replan])

        policy_action = np.asarray(action_plan.popleft(), dtype=np.float32)
        observation, _, done, truncated, info = env.step(convert_action(policy_action))
        steps += 1

        recorder.record(env, obj_name, policy_action, grasp_fn=grasp_fn)

        for key, image in collect_images(observation).items():
            image_queues[key].append(image)
        state_queue.append(observation_to_state(observation))

        success = bool(info.get("success", False))
        if success or done or truncated:
            break

    npz_out: Path | None = None
    if npz_path is not None:
        npz_out = recorder.finalize(
            npz_path,
            mass_kg=condition_physics_dict.get("mass_kg"),
            com_offset_m=condition_physics_dict.get("com_offset_m"),
            com_axis=condition_physics_dict.get("com_axis"),
            seed=seed,
            success=success,
        )

    return success, steps, npz_out


def npz_path_for(output_root: Path, cell: str, condition: str, seed: int) -> Path:
    return output_root / "phase1" / cell / condition / f"ep_{seed}.npz"


def stats_path_for(output_root: Path, cell: str, condition: str) -> Path:
    return output_root / "phase1" / cell / condition / "stats.json"


def update_condition_stats(
    output_root: Path,
    cell: str,
    condition: str,
    horizon: int,
    episode_result: dict[str, Any],
) -> dict[str, Any]:
    """Append ``episode_result`` to ``<cell>/<condition>/stats.json``,
    mirroring ``entry.evaluate_task``'s stats schema (env_name -> cell,
    plus condition/horizon/episodes/success_rate)."""
    path = stats_path_for(output_root, cell, condition)
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            stats = json.load(file)
    else:
        stats = {
            "env_name": cell,
            "condition": condition,
            "horizon": horizon,
            "episodes": [],
        }

    stats["episodes"] = [
        item for item in stats["episodes"] if item["seed"] != episode_result["seed"]
    ]
    stats["episodes"].append(episode_result)
    stats["episodes"].sort(key=lambda item: item["seed"])

    successes = sum(int(item["success"]) for item in stats["episodes"])
    stats["num_episodes"] = len(stats["episodes"])
    stats["successes"] = successes
    stats["success_rate"] = successes / len(stats["episodes"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    return stats


def _probe_mass_kg(
    env: Any, env_name: str, category: str, seed: int, obj_name: str
) -> float:
    """Default-density (probe) mass for whatever mesh ``seed`` samples on
    ``env_name``/``category``, from the per-process ``_PROBE_MASS_CACHE``.

    On a cache hit, this does NOT touch ``env`` at all (no reset) -- the
    caller must already have ``env`` in a state it's happy to proceed from.
    On a miss, performs the probe reset itself (uninstalled density) and
    populates the cache. Deterministic reseed guarantees a fixed
    ``(env_name, category, seed)`` always samples the identical mesh
    (verified in Task 2), so this cache is safe across ALL conditions that
    share a seed, not just repeated calls for the exact same condition.
    """
    key = (env_name, category, seed)
    if key in _PROBE_MASS_CACHE:
        return _PROBE_MASS_CACHE[key]

    env.reset(seed=seed)
    bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
    probe_mass_kg = float(env.sim.model.body_mass[bid])
    _PROBE_MASS_CACHE[key] = probe_mass_kg
    return probe_mass_kg


def run_condition_episode(
    gym: Any,
    env_name: str,
    category: str,
    condition: str,
    seed: int,
    client: PolicyClient,
    horizon: int,
    output_root: Path,
    obj_name: str = DEFAULT_OBJ_NAME,
    com_offset_default: float = 0.02,
    probe_density: float = 100.0,
    mass_tol_pct: float = 2.0,
    com_offset_tol_m: float = 1e-6,
) -> tuple[bool, int, Path]:
    """Compose Task 1's ``condition_physics`` + Task 2's density/CoM
    injection + ``run_episode`` into ONE episode of ONE condition on
    ``env_name``/``category``, writing
    ``output/mass_variation/phase1/<env_name>/<condition>/ep_<seed>.npz``
    and updating that condition's ``stats.json``.

    Both mass AND CoM conditions install the density override (CoM
    conditions carry the "medium" mass level too, per Task 1's
    ``condition_physics`` -- see the module docstring's "Post-review
    fixes" note); the actual live sim state (body_mass, and for CoM
    conditions the body_ipos offset from the authored baseline) is
    asserted against the intended condition right before ``run_episode``
    starts, and THAT measured state -- not the intent -- is what gets
    written into the npz's scalars.
    """
    physics = condition_physics(condition, com_offset_m=com_offset_default)
    is_com_condition = physics["com_offset_m"] != 0.0

    env = gym.make(f"robocasa/{env_name}", split="pretrain", obj_groups=category, seed=seed)
    density_installed = False
    try:
        if is_com_condition:
            # CoM conditions ALWAYS pay exactly 2 resets (module docstring
            # "Reset-count budget"): pass 1 discovers the mesh/authored
            # ipos, the probe mass (cached for later conditions sharing
            # this seed), AND gives this run's own center-CoM settle
            # reference (never skipped, even on a cache hit for the probe
            # mass -- the reference pose is specific to this run's gate
            # check). Pass 2 installs the density override for the medium
            # mass this condition carries (the bug this review found: this
            # branch used to skip density entirely), THEN applies the CoM
            # offset immediately after that SAME reset returns -- no
            # further reset happens (the other bug: run_episode used to
            # reset again on its own, discarding this).
            env.reset(seed=seed)
            bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
            probe_mass_kg = float(env.sim.model.body_mass[bid])
            _PROBE_MASS_CACHE[(env_name, category, seed)] = probe_mass_kg
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
            density_installed = True

            initial_observation, _ = env.reset(seed=seed)
            # Re-resolve bid AFTER this reset (hard_reset=True rebuilds the
            # MuJoCo model, so pass-1's bid is not guaranteed to still be
            # valid/correct) -- mirrors the mass branch's own pattern below.
            # A stray reuse of the pass-1 bid here was a review finding:
            # every other call site in this function re-resolves after its
            # governing reset; this one didn't.
            bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
            apply_com_offset(env, obj_name, physics["com_offset_m"], physics["com_axis"])
            settle_and_gate(env, obj_name, reference_pose=reference_pose)

            axis_idx = axis_index(physics["com_axis"])
            actual_com_offset_m = float(
                env.sim.model.body_ipos[bid][axis_idx] - authored_ipos[axis_idx]
            )
        else:
            # Mass condition: density override, applied BEFORE the reset
            # that matters (settle-consistent -- see module docstring). 1
            # reset in the steady state (cache hit), 2 on a cold start.
            probe_mass_kg = _probe_mass_kg(env, env_name, category, seed, obj_name)
            target_density = mass_to_density(
                physics["mass_kg"], probe_mass_kg, probe_density=probe_density
            )
            install_density_override({obj_name: target_density})
            density_installed = True

            initial_observation, _ = env.reset(seed=seed)
            bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
            actual_com_offset_m = 0.0  # never written for a mass condition.

        # Assert-at-record (CRITICAL-2): verify the LIVE env.sim state
        # matches the intended condition, read fresh right here -- not
        # trusted from `physics` or from an earlier apply_*/install_*
        # call's return value. This is what makes "a reset silently
        # discarded the physics injection" (or "the CoM branch never
        # installed density") loud instead of a wrong-but-plausible npz.
        actual_mass_kg = float(env.sim.model.body_mass[bid])
        if physics["mass_kg"] is not None:
            mass_err_pct = (
                abs(actual_mass_kg - physics["mass_kg"]) / physics["mass_kg"] * 100.0
            )
            assert mass_err_pct < mass_tol_pct, (
                f"{condition} seed={seed}: live body_mass {actual_mass_kg:.4f} kg is "
                f"{mass_err_pct:.2f}% off the intended {physics['mass_kg']:.4f} kg -- "
                "physics injection was not applied (or was discarded) before the "
                "episode started"
            )
        assert abs(actual_com_offset_m - physics["com_offset_m"]) < com_offset_tol_m, (
            f"{condition} seed={seed}: live CoM offset {actual_com_offset_m:.6f} m "
            f"does not match the intended {physics['com_offset_m']:.6f} m"
        )

        measured_physics = {
            "mass_kg": actual_mass_kg,
            "com_offset_m": actual_com_offset_m,
            "com_axis": physics["com_axis"],
        }

        npz_path = npz_path_for(output_root, env_name, condition, seed)
        success, steps, saved_path = run_episode(
            env,
            client,
            measured_physics,
            seed,
            horizon,
            obj_name=obj_name,
            npz_path=npz_path,
            initial_observation=initial_observation,
        )
    finally:
        if density_installed:
            uninstall_density_override()
        env.close()

    update_condition_stats(
        output_root,
        env_name,
        condition,
        horizon,
        {"seed": seed, "success": success, "steps": steps, "npz_path": str(saved_path)},
    )

    return success, steps, saved_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XR1 mass/CoM study: one condition episode.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--server-addr", default="localhost")
    parser.add_argument("--server-port", type=int, default=10086)
    parser.add_argument("--env-name", default=PRIMARY_ENV)
    parser.add_argument("--category", default=PRIMARY_CATEGORY)
    parser.add_argument("--condition", default="MassMedium")
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--n-seeds", type=int, default=35)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--output-root", default="output/mass_variation")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    seeds = episode_seeds(args.base_seed, args.cell_index, args.n_seeds)
    seed = seeds[args.seed_index]
    horizon = args.horizon if args.horizon is not None else get_task_horizon(args.env_name)

    client = XR1SocketClient(
        model_path=args.model_path, host=args.server_addr, port=args.server_port
    )
    try:
        success, steps, npz_path = run_condition_episode(
            gym,
            args.env_name,
            args.category,
            args.condition,
            seed,
            client,
            horizon,
            Path(args.output_root),
        )
    finally:
        client.close()

    logging.info(
        "episode done: condition=%s seed=%s success=%s steps=%s npz=%s",
        args.condition,
        seed,
        success,
        steps,
        npz_path,
    )
    print(
        json.dumps(
            {
                "condition": args.condition,
                "seed": seed,
                "success": success,
                "steps": steps,
                "npz_path": str(npz_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

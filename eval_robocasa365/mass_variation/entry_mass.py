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
conditions install the density override BEFORE the (single) reset that
matters -- density is baked in before ``Kitchen``'s internal settle loop
runs, so one reset is settle-consistent. CoM conditions must reset ONCE to
discover the sampled mesh/authored ipos, THEN reset AGAIN with the SAME
seed (verified deterministic resample) and apply the CoM offset immediately
after that second reset returns, per the settle-loop-runs-inside-reset
finding in Task 2.
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
    install_density_override,
    mass_to_density,
    settle_and_gate,
    uninstall_density_override,
)
from eval_robocasa365.mass_variation.recorder import StepRecorder

PRIMARY_ENV = "PickPlaceCounterToCabinet"
PRIMARY_CATEGORY = "milk"
DEFAULT_OBJ_NAME = "obj"


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
) -> tuple[bool, int, Path | None]:
    """Roll out ONE episode on an already-constructed, already-physics-
    injected ``env`` (see the module docstring's two-pass reset note --
    physics injection happens BEFORE this call, via ``run_condition_episode``
    or a caller doing the same sequence), recording ground truth every step.

    ``condition_physics_dict`` (Task 1's ``condition_physics`` output) is
    NOT applied here: it is folded into the saved npz's scalars (mass_kg,
    com_offset_m, com_axis) so the recorder output is self-describing
    without re-deriving the condition from the cell/seed at analysis time.

    Returns ``(success, steps, npz_path_written_or_None)``.
    """
    from robocasa.utils.env_utils import convert_action

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
) -> tuple[bool, int, Path]:
    """Compose Task 1's ``condition_physics`` + Task 2's density/CoM
    injection + ``run_episode`` into ONE episode of ONE condition on
    ``env_name``/``category``, writing
    ``output/mass_variation/phase1/<env_name>/<condition>/ep_<seed>.npz``
    and updating that condition's ``stats.json``.
    """
    physics = condition_physics(condition, com_offset_m=com_offset_default)
    is_com_condition = physics["com_offset_m"] != 0.0

    env = gym.make(f"robocasa/{env_name}", split="pretrain", obj_groups=category, seed=seed)
    density_installed = False
    try:
        if is_com_condition:
            # Two-pass reset pattern (REQUIRED -- overrides.py module
            # docstring): pass 1 discovers the mesh/authored ipos and gives
            # a center-CoM settle reference; pass 2 re-samples the SAME
            # mesh (same seed) and applies the offset immediately after
            # reset returns.
            env.reset(seed=seed)
            center_settle = settle_and_gate(env, obj_name)
            reference_pose = {"pos": center_settle["post_pos"], "quat": center_settle["post_quat"]}

            env.reset(seed=seed)
            apply_com_offset(env, obj_name, physics["com_offset_m"], physics["com_axis"])
            settle_and_gate(env, obj_name, reference_pose=reference_pose)
        else:
            # Mass condition: density override, applied BEFORE the reset
            # that matters (settle-consistent -- see module docstring).
            env.reset(seed=seed)  # pass 1: discover authored mass at default density.
            bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)
            probe_mass_kg = float(env.sim.model.body_mass[bid])
            if physics["mass_kg"] is not None:
                target_density = mass_to_density(
                    physics["mass_kg"], probe_mass_kg, probe_density=probe_density
                )
                install_density_override({obj_name: target_density})
                density_installed = True
                env.reset(seed=seed)  # pass 2: override active, same seed -> same mesh.

        npz_path = npz_path_for(output_root, env_name, condition, seed)
        success, steps, saved_path = run_episode(
            env,
            client,
            physics,
            seed,
            horizon,
            obj_name=obj_name,
            npz_path=npz_path,
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

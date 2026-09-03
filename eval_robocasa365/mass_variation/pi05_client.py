# Copyright (C) 2026 Xiaomi Corporation.
"""pi0.5 (robocasa365-trained) PolicyClient for the mass/CoM study (Task 7).

Adapts the observation format of the ``robocasa-benchmark/openpi`` fork's
own eval client (``examples/robocasa/main.py`` @ ca4c6d7) behind
``entry_mass.PolicyClient`` so the SAME episode loop / recorder / seeds /
conditions drive both models. Parity facts (file:line refer to that fork):

- **Single-frame observation** (history depth 1): their loop builds the
  server request from the CURRENT ``obs`` only -- no history queue exists
  anywhere in ``main.py`` (the whole element is assembled inline at
  main.py:116-151). Our shared loop still maintains XR1's 4-frame stride-2
  queues; this client consumes only the LAST (current) frame.
- **State 16-D, raw quaternions** (main.py:133-142): concat of
  eef_pos_rel(3), eef_rot_rel quat(4), base_pos(3), base_rot quat(4),
  gripper_qpos(2). NOT XR1's axis-angle 14-D and a different field order,
  hence :func:`state_from_observation` here -- ``run_episode`` uses it for
  the state queue when the client provides it.
- **Cameras** (main.py:118-130, 145-151): agentview_left ->
  ``observation/image``, eye_in_hand -> ``observation/wrist_image``,
  agentview_right -> ``observation/right_image``; each
  ``convert_to_uint8(resize_with_pad(img, 224, 224))``. NO rotation (the
  "rotate 180 degrees" comment at main.py:117 has no corresponding code).
- **Serving**: openpi websocket policy server (``scripts/serve_policy.py``
  + ``WebsocketPolicyServer``); client protocol
  ``openpi_client.websocket_client_policy.WebsocketClientPolicy.infer
  (element)["actions"]`` (main.py:94,154).
- **Chunk/replan**: model ``action_horizon=50`` (Pi0Config default,
  ``src/openpi/models/pi0_config.py:26``; the ``pi05_pretrain_human300``
  config at ``src/openpi/training/config.py:1274-1295`` does not override
  it); server-side ``RobocasaOutputs`` slices actions to
  ``[:, :12]`` (``src/openpi/policies/robocasa_policy.py:127``); the
  fork's eval replans every ``replan_steps=5`` steps (main.py:36).

The pure pieces (:func:`state_from_observation`, :func:`pack_pi05_element`,
:func:`slice_action_chunk`) import only numpy + ``openpi_client``'s pure
image tools; the websocket import is deferred to ``Pi05Client.__init__``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from openpi_client import image_tools

PI05_RESIZE_SIZE = 224  # main.py:35 (Args.resize_size default)
PI05_REPLAN_STEPS = 5  # main.py:36 (Args.replan_steps default)
PI05_CHUNK_LEN = 50  # Pi0Config.action_horizon default (pi0_config.py:26)
PI05_ACTION_DIM = 12  # RobocasaOutputs slice (robocasa_policy.py:127)

# element key -> our observation/camera-queue key (entry.CAMERA_KEYS names).
PI05_CAMERA_MAP = {
    "observation/image": "video.robot0_agentview_left",
    "observation/wrist_image": "video.robot0_eye_in_hand",
    "observation/right_image": "video.robot0_agentview_right",
}

_STATE_FIELDS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)


def state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    """The fork's 16-D proprio vector (main.py:133-142): raw quaternions,
    gripper last. Used by ``run_episode`` for the state queue in place of
    XR1's axis-angle ``observation_to_state``."""
    state = np.concatenate(
        [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in _STATE_FIELDS]
    ).astype(np.float32)
    if state.shape != (16,):
        raise ValueError(f"Expected a 16-D pi0.5 state, got {state.shape}")
    return state


def _last_frame(stacked: np.ndarray) -> np.ndarray:
    """Current frame from a stacked (history, ...) array -- ``run_episode``'s
    ``sample_history`` puts the newest frame LAST."""
    arr = np.asarray(stacked)
    return arr[-1]


def pack_pi05_element(
    obs_history: dict[str, Any],
    instruction: str,
    resize_size: int = PI05_RESIZE_SIZE,
) -> dict[str, Any]:
    """Build the fork's inference element (main.py:145-151) from the shared
    loop's ``obs_history`` (state (H,16) + per-camera (H,256,256,3) stacks),
    consuming ONLY the last (current) frame -- pi0.5 is single-frame."""
    element: dict[str, Any] = {}
    for element_key, camera_key in PI05_CAMERA_MAP.items():
        frame = np.ascontiguousarray(_last_frame(obs_history["images"][camera_key]))
        element[element_key] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(frame, resize_size, resize_size)
        )
    element["observation/state"] = np.asarray(_last_frame(obs_history["state"]), dtype=np.float32)
    element["prompt"] = instruction
    return element


def slice_action_chunk(actions: np.ndarray, action_dim: int = PI05_ACTION_DIM) -> np.ndarray:
    """Defensive mirror of the server-side ``RobocasaOutputs`` slice
    (robocasa_policy.py:127): keep the first ``action_dim`` dims."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] < action_dim:
        raise ValueError(
            f"Expected an (N, >= {action_dim}) action chunk, got {actions.shape}"
        )
    return np.asarray(actions[:, :action_dim], dtype=np.float32)


class Pi05Client:
    """``entry_mass.PolicyClient`` over the openpi websocket server.

    ``chunk_len`` starts at the config-derived 50 and is refreshed from
    every actual server response (``run_episode`` reads it AFTER ``infer``),
    so a checkpoint with a different action_horizon fails loudly only if it
    returns fewer than ``replan`` actions.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        replan: int = PI05_REPLAN_STEPS,
        resize_size: int = PI05_RESIZE_SIZE,
    ) -> None:
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.chunk_len = PI05_CHUNK_LEN
        self.replan = replan
        self.resize_size = resize_size

    # run_episode uses this (when present) instead of XR1's
    # observation_to_state to fill the state queue.
    state_from_observation = staticmethod(state_from_observation)

    def infer(self, obs_history: dict[str, Any], instruction: str) -> np.ndarray:
        element = pack_pi05_element(obs_history, instruction, self.resize_size)
        actions = slice_action_chunk(self._client.infer(element)["actions"])
        self.chunk_len = len(actions)
        return actions

    def close(self) -> None:
        ws = getattr(self._client, "_ws", None)
        if ws is not None:
            ws.close()

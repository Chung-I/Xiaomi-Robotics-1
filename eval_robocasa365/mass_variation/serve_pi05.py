# Copyright (C) 2026 Xiaomi Corporation.
"""Thin pi0.5 policy-server launcher for the mass/CoM study (Task 7).

Runs under the ``~/Codes/openpi-robocasa`` venv (JAX), NOT the robocasa or
mibot venvs. Mirrors the fork's own ``scripts/serve_policy.py`` (policy:
checkpoint route -> ``create_trained_policy`` -> ``WebsocketPolicyServer``)
with ONE deviation, the reason this wrapper exists: the training config's
``LeRobotRobocasaDataConfig.data_dirs`` lists the (absent) training
datasets, and ``create()`` falls back to reading their ``meta/stats.json``
for norm stats when the config assets are missing
(``src/openpi/training/config.py`` LeRobotRobocasaDataConfig.create),
which raises FileNotFoundError on a machine without the datasets. We clear
``data_dirs`` so ``create_trained_policy`` takes its documented serving
path instead: norm stats from ``<checkpoint>/assets/norm_stats.json``
(``src/openpi/policies/policy_config.py:66-72``), exactly the stats the
checkpoint shipped with.

Usage (from run_phase1.build_server_launch):
    <openpi-venv>/bin/python -u serve_pi05.py \
        --config pi05_pretrain_human300 --dir <ckpt>/75000 --port 8000
"""

from __future__ import annotations

import argparse
import dataclasses
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="pi05_pretrain_human300")
    parser.add_argument("--dir", required=True, help="Checkpoint directory (contains params/ and assets/).")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    # Heavy imports after arg parsing (JAX + full robocasa env-zoo import).
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    # Clear the training data_dirs (see module docstring): serving needs no
    # datasets; norm stats come from <checkpoint>/assets.
    train_config = dataclasses.replace(
        train_config, data=dataclasses.replace(train_config.data, data_dirs=[])
    )

    policy = _policy_config.create_trained_policy(train_config, args.dir)
    logging.info(
        "pi0.5 policy loaded from %s (config %s); serving on port %d",
        args.dir, args.config, args.port,
    )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Xiaomi Corporation.
"""Phase-1 benchmark driver for the XR1 mass/CoM study (Plan 1, Task 6,
under Plan amendment A).

Modes (both on the ONE primary cell, matched seeds via ``episode_seeds``):

- ``sanity``  -- 3 mass conditions x 5 seeds  (15 episodes). Writes
  ``output/mass_variation/sanity_sweep_<model>.json`` with the amendment's
  gate: MassLight (0.15 kg) success rate must be within 0.2 of the model's
  published per-task baseline.
- ``phase1``  -- 3 mass conditions x 35 seeds (105 episodes). Writes
  ``output/mass_variation/phase1/metrics_<model>.csv`` (tidy per-condition
  metrics, ``model`` column) + ``phase1_summary_<model>.json``.

Sanity seeds are the first 5 of the 35 phase-1 seeds and share the same
npz paths, so sanity episodes are reused by phase1 (both run under the
same fresh-server-per-batch rule below).

Operational rule (ledgered, binding): every condition batch starts with a
FRESH XR1 server -- the ledger records a stale server RNG stream producing
a full-horizon failure smoke. This driver kills the server listening on
``--port`` by its EXACT pid (parsed from ``ss -ltnp``; never pkill -f),
relaunches ``deploy/server.py`` detached with ``MIBOT_SERVER_SEED=7``
(mirroring scripts/deploy.sh's invocation: ``python -u deploy/server.py
--model <ckpt> --port <port>`` with ``TOKENIZERS_PARALLELISM=false``,
using ``.venv-mibot``'s python), and confirms the log grows and the port
listens before running episodes. Each restart is recorded (start time,
pid, seed) in ``server_restarts_<model>.json``. A resumed invocation that
continues a partially-done batch also restarts the server first --
conservative, and consistent with the rule.

Resumable: episodes whose npz already exists are skipped, so bounded
foreground invocations (``--time-budget-s``) can be repeated until the
mode completes; the completing invocation computes metrics, writes the
summary/CSV, and logs the wandb run (project ``mass-com-xr1``, run
``<mode>-<model>``) -- wandb is only logged once per mode unless
``--rewandb``.

Run under the robocasa venv with EGL:
  MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python -m \
      eval_robocasa365.mass_variation.run_phase1 --model xr1 --mode sanity
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import signal
import socket as _socket
import subprocess
import time
from pathlib import Path

from eval_robocasa365.mass_variation import metrics
from eval_robocasa365.mass_variation.conditions import MASS_LEVELS_KG, episode_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_PYTHON = REPO_ROOT / ".venv-mibot" / "bin" / "python"
SERVER_CHECKPOINT = REPO_ROOT / "checkpoints" / "Xiaomi-Robotics-1-RoboCasa365"
SERVER_SEED = 7

# ---- pi0.5 comparison arm (Task 7) --------------------------------------
OPENPI_ROOT = Path.home() / "Codes" / "openpi-robocasa"
OPENPI_PYTHON = OPENPI_ROOT / ".venv" / "bin" / "python"
PI05_CHECKPOINT = (
    OPENPI_ROOT / "checkpoints" / "pi05_pretrain_human300" / "multitask_learning" / "75000"
)
PI05_TRAIN_CONFIG = "pi05_pretrain_human300"

# Published per-task baseline (None = no per-task number published; the
# sanity gate then uses ABSOLUTE_GATE_THRESHOLD -- the design addendum's
# decision rule: pi0.5 MassLight success < 0.25 -> STOP/BLOCKED).
PUBLISHED_BASELINE = {"xr1": 0.86, "pi05_robocasa": None}
ABSOLUTE_GATE_THRESHOLD = {"pi05_robocasa": 0.25}
BASELINE_TOLERANCE = 0.2
MASS_CONDITIONS = ("MassLight", "MassMedium", "MassHeavy")
MODE_SEEDS = {"sanity": 5, "phase1": 35}
# One model server on the GPU at a time: each model's default port; the
# driver kills whatever listens on the OTHER models' ports before starting.
MODEL_PORTS = {"xr1": 10086, "pi05_robocasa": 8000}
# wandb run names per the plan: sanity-pi05 / phase1-pi05 (not the full
# model id), sanity-xr1 / phase1-xr1 as in T6.
WANDB_MODEL_NAME = {"xr1": "xr1", "pi05_robocasa": "pi05"}


def cell_dir_for(env_name: str, model: str) -> str:
    """npz/stats directory name under ``phase1/`` for one model's episodes.
    XR1 keeps the unsuffixed T6 pathing (its 105 npz stay untouched);
    every other model gets a ``__<model>`` suffix."""
    return env_name if model == "xr1" else f"{env_name}__{model}"

log = logging.getLogger("run_phase1")


# --------------------------------------------------------------------------
# Server lifecycle (operational rule)
# --------------------------------------------------------------------------

def server_pid(port: int) -> int | None:
    """Exact pid of the process LISTENing on ``port`` via ``ss -ltnp``."""
    out = subprocess.run(
        ["ss", "-ltnp"], capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        if re.search(rf"[:\]]{port}\b", line.split()[3] if len(line.split()) > 3 else ""):
            match = re.search(r"pid=(\d+)", line)
            if match:
                return int(match.group(1))
    return None


def _port_listening(port: int, host: str = "127.0.0.1") -> bool:
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait(predicate, timeout_s: float, poll_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def kill_server(port: int, timeout_s: float = 30.0) -> int | None:
    """SIGTERM (then SIGKILL) the exact pid on ``port``; wait for the port
    to free up. Returns the killed pid (or None if nothing listened)."""
    pid = server_pid(port)
    if pid is None:
        return None
    log.info("Killing server pid=%d on port %d (SIGTERM)", pid, port)
    os.kill(pid, signal.SIGTERM)
    if not _wait(lambda: server_pid(port) is None, timeout_s):
        log.warning("Server pid=%d survived SIGTERM; sending SIGKILL", pid)
        os.kill(pid, signal.SIGKILL)
        if not _wait(lambda: server_pid(port) is None, timeout_s):
            raise RuntimeError(f"Port {port} still occupied after SIGKILL of {pid}")
    return pid


def build_server_launch(model: str, port: int) -> tuple[list[str], dict, str]:
    """(cmd, extra_env, cwd) to launch one model's policy server.

    - xr1: stock pickle server (deploy/server.py), MIBOT_SERVER_SEED=7,
      mirrors scripts/deploy.sh (T6 rule).
    - pi05_robocasa: openpi websocket server via this repo's thin
      serve_pi05.py launcher, run with the openpi-robocasa venv's python
      (JAX); XLA_PYTHON_CLIENT_MEM_FRACTION leaves headroom for the EGL
      sim process on the shared GPU.
    """
    if model == "xr1":
        cmd = [
            str(SERVER_PYTHON), "-u", "deploy/server.py",
            "--model", str(SERVER_CHECKPOINT), "--port", str(port),
        ]
        return cmd, {
            "MIBOT_SERVER_SEED": str(SERVER_SEED),
            "TOKENIZERS_PARALLELISM": "false",
        }, str(REPO_ROOT)
    if model == "pi05_robocasa":
        cmd = [
            str(OPENPI_PYTHON), "-u",
            str(REPO_ROOT / "eval_robocasa365" / "mass_variation" / "serve_pi05.py"),
            "--config", PI05_TRAIN_CONFIG,
            "--dir", str(PI05_CHECKPOINT),
            "--port", str(port),
        ]
        return cmd, {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.7"}, str(OPENPI_ROOT)
    raise ValueError(f"Unknown model {model!r}")


def start_server(
    port: int,
    log_path: Path,
    model: str = "xr1",
    timeout_s: float = 600.0,
) -> dict:
    """Launch a fresh detached server (own session, nohup-equivalent) and
    block until the log grows AND the port listens. Returns a restart
    record for the run metadata."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd, extra_env, cwd = build_server_launch(model, port)
    env = dict(os.environ)
    env.update(extra_env)
    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    with open(log_path, "ab") as log_file:
        log_file.write(f"\n===== run_phase1 restart {started_at} =====\n".encode())
        log_file.flush()
        size0 = log_file.tell()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log.info("Launched server pid=%d; waiting for port %d ...", proc.pid, port)

    def _up() -> bool:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server pid={proc.pid} exited early (code {proc.returncode}); "
                f"see {log_path}"
            )
        return _port_listening(port) and log_path.stat().st_size > size0

    if not _wait(_up, timeout_s, poll_s=2.0):
        raise RuntimeError(
            f"Server pid={proc.pid} did not come up on port {port} within "
            f"{timeout_s:.0f}s; see {log_path}"
        )
    listening_at = _dt.datetime.now().isoformat(timespec="seconds")
    log.info("Server up: pid=%d port=%d (listening at %s)", proc.pid, port, listening_at)
    return {
        "pid": proc.pid,
        "port": port,
        "cmd": cmd,
        "started_at": started_at,
        "listening_at": listening_at,
        "server_seed": SERVER_SEED if model == "xr1" else None,
        "server_model": model,
        "log": str(log_path),
    }


def record_restart(restarts_path: Path, record: dict) -> None:
    restarts_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if restarts_path.exists():
        with restarts_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
    records.append(record)
    with restarts_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["xr1", "pi05_robocasa"], default="xr1")
    parser.add_argument("--mode", choices=["sanity", "phase1"], required=True)
    parser.add_argument("--conditions", nargs="+", default=list(MASS_CONDITIONS))
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Seed count override (default: 5 sanity / 35 phase1).",
    )
    parser.add_argument("--env-name", default="PickPlaceCounterToCabinet")
    parser.add_argument("--category", default="milk")
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--output-root", default="output/mass_variation")
    parser.add_argument(
        "--port", type=int, default=None,
        help="Server port (default: the model's own port, 10086 xr1 / 8000 pi05).",
    )
    parser.add_argument(
        "--client-model-path", default=None,
        help="Processor path for the socket client (default: entry.DEFAULT_MODEL_PATH).",
    )
    parser.add_argument(
        "--time-budget-s", type=float, default=None,
        help="Exit cleanly (resumable) once this much wall time has elapsed; "
        "checked between episodes and before server restarts.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--rewandb", action="store_true",
        help="Log wandb even if this mode's summary json already exists.",
    )
    parser.add_argument(
        "--no-server-restart", action="store_true",
        help="DEBUG ONLY: reuse the running server (violates the batch rule).",
    )
    return parser.parse_args(argv)


def summary_path_for(output_root: Path, mode: str, model: str) -> Path:
    if mode == "sanity":
        return output_root / f"sanity_sweep_{model}.json"
    return output_root / "phase1" / f"phase1_summary_{model}.json"


def evaluate_sanity_gate(df, model: str) -> dict:
    """Amendment-A gate. With a published per-task baseline: MassLight
    success within BASELINE_TOLERANCE of it. Without one (pi0.5), the
    design addendum's absolute decision rule: success < 0.25 -> STOP."""
    baseline = PUBLISHED_BASELINE[model]
    if baseline is not None:
        threshold = baseline - BASELINE_TOLERANCE
    else:
        threshold = ABSOLUTE_GATE_THRESHOLD[model]
    light = df[df["condition"] == "MassLight"].iloc[0]
    light_rate = float(light["success_rate"])
    return {
        "published_baseline": baseline,
        "tolerance": BASELINE_TOLERANCE if baseline is not None else None,
        "threshold": threshold,
        "mass_light_success_rate": light_rate,
        "passed": bool(light_rate >= threshold - 1e-9),
    }


def log_wandb(mode: str, model: str, df, config: dict, gate: dict | None) -> str:
    import wandb

    run = wandb.init(
        project="mass-com-xr1",
        name=f"{mode}-{WANDB_MODEL_NAME.get(model, model)}",
        job_type=mode,
        config=config,
        reinit=True,
    )
    table = wandb.Table(dataframe=df)
    payload: dict = {"per_condition_metrics": table}
    for _, row in df.iterrows():
        cond = row["condition"]
        for key in (
            "success_rate", "grasp_rate", "lift_rate",
            "t_success_mean_s", "drop_after_lift_rate",
        ):
            payload[f"{key}/{cond}"] = row[key]
    if gate is not None:
        payload["sanity_gate/mass_light_success_rate"] = gate["mass_light_success_rate"]
        payload["sanity_gate/threshold"] = gate["threshold"]
        payload["sanity_gate/passed"] = int(gate["passed"])
    run.log(payload)
    url = run.url
    run.finish()
    return url


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    t0 = time.monotonic()

    def budget_left() -> bool:
        return args.time_budget_s is None or (time.monotonic() - t0) < args.time_budget_s

    n_seeds = args.seeds if args.seeds is not None else MODE_SEEDS[args.mode]
    seeds = episode_seeds(args.base_seed, args.cell_index, n_seeds)
    port = args.port if args.port is not None else MODEL_PORTS[args.model]
    cell_dir = cell_dir_for(args.env_name, args.model)
    output_root = Path(args.output_root)
    phase1_root = output_root / "phase1"
    restarts_path = phase1_root / f"server_restarts_{args.model}.json"
    runlog_path = phase1_root / f"runlog_{args.model}.jsonl"
    server_log = output_root / "logs" / f"server_{port}.log"

    # Heavy imports only when we may actually run episodes.
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from eval_robocasa365.entry import DEFAULT_MODEL_PATH
    from eval_robocasa365.mass_variation.entry_mass import (
        XR1SocketClient,
        npz_path_for,
        run_condition_episode,
    )

    horizon = args.horizon if args.horizon is not None else get_task_horizon(args.env_name)
    client_model_path = args.client_model_path or DEFAULT_MODEL_PATH

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    ).stdout.strip()

    pending = {
        condition: [
            seed for seed in seeds
            if not npz_path_for(output_root, cell_dir, condition, seed).exists()
        ]
        for condition in args.conditions
    }
    total_pending = sum(len(v) for v in pending.values())
    log.info(
        "mode=%s model=%s cell=%s cell_dir=%s horizon=%d seeds=%d pending=%d/%d",
        args.mode, args.model, args.env_name, cell_dir, horizon, n_seeds,
        total_pending, n_seeds * len(args.conditions),
    )

    # One model server on the GPU at a time (T7 rule): before any batch,
    # kill whatever listens on the OTHER models' ports -- by exact pid via
    # ss -ltnp, same mechanism as the per-batch restart below.
    if total_pending > 0 and not args.no_server_restart:
        for other_model, other_port in MODEL_PORTS.items():
            if other_model != args.model and other_port != port:
                killed = kill_server(other_port)
                if killed is not None:
                    log.info(
                        "Killed %s server pid=%d on port %d (one model on "
                        "the GPU at a time)", other_model, killed, other_port,
                    )
                    record_restart(restarts_path, {
                        "killed_other_model": other_model, "pid": killed,
                        "port": other_port, "model": args.model,
                        "at": _dt.datetime.now().isoformat(timespec="seconds"),
                    })

    def make_client():
        if args.model == "xr1":
            return XR1SocketClient(
                model_path=client_model_path, host="localhost", port=port
            )
        from eval_robocasa365.mass_variation.pi05_client import Pi05Client

        return Pi05Client(host="127.0.0.1", port=port)

    budget_exit = False
    for condition in args.conditions:
        todo = pending[condition]
        if not todo:
            log.info("%s: complete (all %d npz present), skipping", condition, n_seeds)
            continue
        if not budget_left():
            budget_exit = True
            break

        if args.no_server_restart:
            log.warning("%s: --no-server-restart set; reusing running server", condition)
        else:
            kill_server(port)
            record = start_server(port, server_log, model=args.model)
            record.update(
                {"mode": args.mode, "condition": condition, "model": args.model,
                 "pending_seeds": todo}
            )
            record_restart(restarts_path, record)

        client = make_client()
        try:
            for seed in todo:
                if not budget_left():
                    budget_exit = True
                    break
                ep_t0 = time.monotonic()
                success, steps, npz_path = run_condition_episode(
                    gym, args.env_name, args.category, condition, seed,
                    client, horizon, output_root, cell_dir=cell_dir,
                )
                wall_s = time.monotonic() - ep_t0
                log.info(
                    "%s seed=%d: success=%s steps=%d wall=%.1fs",
                    condition, seed, success, steps, wall_s,
                )
                runlog_path.parent.mkdir(parents=True, exist_ok=True)
                with runlog_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps({
                        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
                        "mode": args.mode, "condition": condition, "seed": seed,
                        "success": success, "steps": steps,
                        "wall_s": round(wall_s, 1), "npz": str(npz_path),
                    }) + "\n")
        finally:
            client.close()
        if budget_exit:
            break

    remaining = {
        condition: [
            seed for seed in seeds
            if not npz_path_for(output_root, cell_dir, condition, seed).exists()
        ]
        for condition in args.conditions
    }
    n_remaining = sum(len(v) for v in remaining.values())
    if n_remaining > 0:
        log.info(
            "BUDGET_EXIT: %d episodes remaining (%s); re-invoke to resume.",
            n_remaining,
            {k: len(v) for k, v in remaining.items() if v},
        )
        print(json.dumps({"status": "incomplete", "remaining": n_remaining}))
        return 0

    # ---- Mode complete: metrics, summary, wandb -------------------------
    df = metrics.metrics_dataframe(
        phase1_root, cell=args.env_name, conditions=args.conditions,
        model=args.model, cell_dir=cell_dir,
    )
    if args.mode == "phase1":
        csv_path = metrics.write_csv(df, phase1_root / f"metrics_{args.model}.csv")
        log.info("Wrote %s", csv_path)

    gate = evaluate_sanity_gate(df, args.model) if args.mode == "sanity" else None

    summary_path = summary_path_for(output_root, args.mode, args.model)
    already_summarized = summary_path.exists()

    restarts = []
    if restarts_path.exists():
        with restarts_path.open("r", encoding="utf-8") as file:
            restarts = json.load(file)

    config = {
        "model": args.model,
        "mode": args.mode,
        "cell": args.env_name,
        "cell_dir": cell_dir,
        "category": args.category,
        "conditions": args.conditions,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "base_seed": args.base_seed,
        "cell_index": args.cell_index,
        "horizon": horizon,
        "mass_levels_kg": MASS_LEVELS_KG,
        "server_seed": SERVER_SEED if args.model == "xr1" else None,
        "server_checkpoint": str(
            SERVER_CHECKPOINT if args.model == "xr1" else PI05_CHECKPOINT
        ),
        "git_sha": git_sha,
        "control_hz": metrics.CONTROL_HZ,
        "drop_threshold_m": metrics.DROP_M,
        "drop_window_steps": metrics.DROP_WINDOW_STEPS,
    }

    wandb_url = None
    if not args.no_wandb and (args.rewandb or not already_summarized):
        wandb_url = log_wandb(args.mode, args.model, df, config, gate)
        log.info("wandb run: %s", wandb_url)

    summary = {
        "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "per_condition": df.to_dict(orient="records"),
        "sanity_gate": gate,
        "wandb_url": wandb_url,
        "server_restarts": [
            r for r in restarts if r.get("mode") == args.mode
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    log.info("Wrote %s", summary_path)

    print(df.to_string(index=False))
    if gate is not None:
        verdict = "PASS" if gate["passed"] else "FAIL"
        if gate["published_baseline"] is not None:
            basis = (
                f"(baseline {gate['published_baseline']:.2f} - "
                f"{gate['tolerance']:.1f})"
            )
        else:
            basis = "(absolute decision-rule threshold; no published per-task baseline)"
        print(
            f"SANITY GATE {verdict}: MassLight success "
            f"{gate['mass_light_success_rate']:.3f} vs threshold "
            f"{gate['threshold']:.2f} {basis}"
        )
    print(json.dumps({"status": "complete", "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

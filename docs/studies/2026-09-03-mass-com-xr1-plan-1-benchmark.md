# XR1 Mass/CoM Plan 1: Variation Infra + Calibration + Behavioral Benchmark

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hidden mass/CoM variation machinery for robocasa365, per-step F/T ground-truth recording, a deterministic in-process XR1 runner, calibrated mass levels, and the Phase-1 behavioral benchmark (500 episodes).

**Architecture:** All study code lives in `eval_robocasa365/mass_variation/` in this repo; robocasa/robosuite stay unedited (monkeypatch at the sampling seam, post-reset MuJoCo writes for CoM). Behavioral runs use the stock pickle server; the offline runner loads the model in-process for later plans and gates determinism now.

**Tech Stack:** sim side runs in the robocasa venv (`~/Codes/robocasa/.venv/bin/python`, mujoco 3.3.1); model server in `.venv-mibot`; offline runner in `.venv-mibot`. Tests: `~/Codes/robocasa/.venv/bin/python -m pytest eval_robocasa365/mass_variation -v` (pure parts import no sim). wandb project `mass-com-xr1`.

**Spec:** `docs/studies/2026-09-03-mass-com-xr1-design.md` (binding; carries the ported metrics/certificate/window rules).

## Global Constraints

- NEVER push to `origin` (upstream XiaomiRobotics). Commit locally on `study/mass-com-xr1`; the controller pushes to the user's fork once added.
- robocasa (`~/Codes/robocasa`) and robosuite (`~/Codes/robosuite`) are read-only for this study — no edits, no new files. Injection is done by monkeypatching from study code.
- Verified seams (from the 2026-09-03 exploration; treat as ground truth): density kwarg reaches `MJCFObject` via `robocasa/utils/env_utils.py:1488`; sampling seam `Kitchen.sample_object` → `sample_kitchen_object`; CoM writes = `sim.model.body_ipos[bid]` + `body_inertia[bid]` AFTER `env.reset()` returns but the settle loop runs inside reset (`kitchen.py:1146`) — so CoM changes require the resample-then-forward pattern in Task 2, NOT naive post-reset writes; body ids re-resolved per episode via `sim.model.body_name2id(env.objects[name].root_body)`; per-episode reseed (`env.reset(seed=...)` via the gym wrapper) is REQUIRED for matched pairs; wrist F/T = `env.robots[0].ee_force/.ee_torque`; object wrench = `sim.data.cfrc_ext[bid]`; grasp = `robocasa.utils.object_utils.check_obj_grasped(env, obj_name)`.
- Determinism policy: behavioral phase = stock server, `MIBOT_SERVER_SEED=7`, one worker per run, worker→port mapping logged. Offline runner = in-process, explicit noise, bit-exact gate.
- Episode seeds: `seed_list = [args.seed + 1000*cell_index + k for k in range(50)]`, IDENTICAL across conditions of a cell (matched pairs by construction).
- Every run logs to wandb project `mass-com-xr1` (job_type per phase) with its full config.
- pytest always `-v`. Commits after green with the standard trailer. `MUJOCO_GL=egl` for all sim processes. Never `pgrep -f`/`pkill -f` (use `ps -eo cmd | grep -c "[x]name"`).

---

### Task 1: condition table + preflight validator

**Files:**
- Create: `eval_robocasa365/mass_variation/__init__.py` (empty), `eval_robocasa365/mass_variation/conditions.py`
- Create: `eval_robocasa365/mass_variation/preflight.py` (script)
- Test: `eval_robocasa365/mass_variation/test_conditions.py`

**Interfaces:**
- Produces: `CONDITIONS = ("MassLight", "MassMedium", "MassHeavy", "CoMOffA", "CoMOffB")`;
  `condition_physics(condition, knee_kg, com_offset_m) -> dict(mass_kg | None, com_offset_m, com_axis)` — mass conditions return `mass_kg = {0.3,1.0,1.7}*knee` and zero CoM offset; CoM conditions return knee mass and ±`com_offset_m` on axis `"y"` (per-cell axis finalized in Task 2's settle gate);
  `episode_seeds(base_seed, cell_index, n=50) -> list[int]` per the Global Constraints formula;
  `load_mass_levels(path) -> dict` (repo-root anchored, FileNotFoundError on missing).

- [ ] **Step 1: failing tests** — `condition_physics` ratios exact (0.3/1.0/1.7), CoM conditions carry knee mass, seeds formula matches spec and is identical across conditions, loader raises on missing path.

```python
def test_mass_ratios_and_com_conditions():
    p = condition_physics("MassHeavy", knee_kg=0.6, com_offset_m=0.02)
    assert p["mass_kg"] == pytest.approx(1.02) and p["com_offset_m"] == 0.0
    q = condition_physics("CoMOffA", knee_kg=0.6, com_offset_m=0.02)
    assert q["mass_kg"] == pytest.approx(0.6) and q["com_offset_m"] == 0.02

def test_seeds_matched_across_conditions():
    assert episode_seeds(7, 1, 3) == [1007, 1008, 1009]
```

- [ ] **Step 2:** RED with `~/Codes/robocasa/.venv/bin/python -m pytest eval_robocasa365/mass_variation/test_conditions.py -v` → implement (~60 lines, no sim imports) → GREEN.
- [ ] **Step 3: preflight script** (sim-touching, assertions inline): for each candidate cell — (`PickPlaceCounterToCabinet`, `milk`) plus (`PickPlaceToasterToCounter`, each of `["bottled_water","boxed_food","bottled_drink","boxed_drink","canned_food"]`) — construct the env with `obj_groups=<category>` via `gym.make(..., split="pretrain")` + reset with 3 seeds; assert the pool is non-empty (no `ValueError`), record sampled mesh names, default masses (`sim.model.body_mass[bid]`), and horizon. Write `output/mass_variation/preflight.json`. Pick the secondary category = the first that passes with ≥5 distinct meshes over 10 seeds; print the choice.
- [ ] **Step 4:** run preflight (`MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python eval_robocasa365/mass_variation/preflight.py`), commit code + preflight.json (force-add; `output/` may be gitignored).

### Task 2: physics injection (density monkeypatch + CoM writer)

**Files:**
- Create: `eval_robocasa365/mass_variation/overrides.py`
- Create: `eval_robocasa365/mass_variation/verify_overrides.py` (script)
- Test: `eval_robocasa365/mass_variation/test_overrides.py` (pure parts)

**Interfaces:**
- Produces:
  - `mass_to_density(target_mass_kg, probe_mass_kg, probe_density=100.0) -> float` (pure; linear, verified 10×→10×).
  - `install_density_override(density_by_objname: dict)` — monkeypatches `robocasa.utils.env_utils.sample_kitchen_object` (the function `Kitchen.sample_object` delegates to): wraps the original, sets `object_kwargs["density"]` for the target object name before `MJCFObject` construction. `uninstall_density_override()` restores.
  - `apply_com_offset(env, obj_name, offset_m, axis) -> dict` — resolves `bid = env.sim.model.body_name2id(env.objects[obj_name].root_body)`; writes `body_ipos[bid][axis_idx] += offset_m` (absolute from the authored value, idempotent: stores authored on first call keyed by mesh id); calls `env.sim.forward()`; returns `{bid, mass, ipos}`.
  - `settle_and_gate(env, obj_name, n_steps=20, pos_tol_m=1e-3, rot_tol_deg=0.5) -> dict` — steps the env with zero actions, compares object pose to a center-CoM reference episode; the CoM-visibility gate from the design.
- Two-pass reset pattern (documented in the module docstring, REQUIRED because the settle loop runs inside `reset()`): for CoM conditions, reset once to discover the sampled mesh + authored physics, then reset AGAIN with the same seed (identical sample, verified property), apply `apply_com_offset` immediately after the second reset returns, then run `settle_and_gate`. Mass conditions need only the density override (applied at sampling, inertia-consistent, settle-safe).

- [ ] **Step 1:** pure tests: `mass_to_density` linearity; axis-index mapping; idempotency of the authored-value bookkeeping (fake model object with numpy arrays).
- [ ] **Step 2:** RED → implement (~120 lines) → GREEN.
- [ ] **Step 3: verification script** on the primary cell, printed table + assertions: (a) density route: request 0.60 kg → measured `body_mass` within 1%; same mesh + same `body_ipos` as the 100-density baseline at the same seed; (b) CoM route: offset ±0.02 m on y → `body_ipos` moved exactly, mass unchanged, `data.xipos` propagates; settle gate passes (pose delta < 1 mm/0.5°) or the script prints the failing axis and tries `x` (axis decision recorded in output); (c) matched-pair check: two conditions at the same seed sample the identical mesh (name equality asserted); (d) `cfrc_ext` Fz ≈ −mg within 5% after settle at 0.60 kg.
- [ ] **Step 4:** run it, commit code + `output/mass_variation/verify_overrides.json`.

### Task 3: ground-truth recorder + study entry loop

**Files:**
- Create: `eval_robocasa365/mass_variation/recorder.py`, `eval_robocasa365/mass_variation/entry_mass.py`
- Test: `eval_robocasa365/mass_variation/test_recorder.py`

**Interfaces:**
- Consumes: Tasks 1–2; `eval_robocasa365/entry.py`'s `EvalClient`, `reset_env`, `convert_action` (imported, not copied, where importable; the episode loop itself is a rewritten ~80-line function `run_episode(env, client, condition_physics_dict, seed, horizon) -> (success, steps, npz_path)` because the stock loop has no per-step hook).
- Produces per episode: `output/mass_variation/phase1/<cell>/<condition>/ep_<seed>.npz` with per-step arrays `ee_force (T,3), ee_torque (T,3), cfrc_obj (T,6), obj_pos (T,3), obj_quat (T,4), gripper_qpos (T,2), grasped (T,), actions (T,12), eef_pos (T,3), eef_rot (T,3)` and scalars `mass_kg, com_offset_m, com_axis, seed, success, liftoff_step` (first step with `grasped` AND z-rise ≥ 0.05 m; −1 if never); plus `stats.json` per condition mirroring the stock schema.
- `recorder.py` is pure-ish: `StepRecorder.record(env, obj_name, action)` pulls the arrays (env access isolated in one method), `finalize(path, **scalars)`; liftoff logic `liftoff_step(z, grasped, rise_m=0.05) -> int` is pure.

- [ ] **Step 1:** pure tests: `liftoff_step` (no-lift → −1; rise while not grasped doesn't count; boundary exact); recorder shape/finalize round-trip with a fake env stub.
- [ ] **Step 2:** RED → implement → GREEN.
- [ ] **Step 3: smoke episode** — one full episode on the primary cell, MassMedium-at-default-density, stock server running (`scripts/deploy.sh` port 10086, `MIBOT_SERVER_SEED=7`): assert npz written with T == steps, grasped/liftoff plausible, video optional off. Commit.

### Task 4: offline in-process runner + determinism gate

**Files:**
- Create: `eval_robocasa365/mass_variation/xr1_runner.py`
- Test: `eval_robocasa365/mass_variation/test_xr1_runner.py` (pure parts: message building parity)

**Interfaces:**
- Produces: `XR1Runner(checkpoint_dir, device)` — loads `AutoModel.from_pretrained(..., trust_remote_code=True, torch_dtype=bfloat16)` in `.venv-mibot`; `infer(frames_by_cam: dict[str, list[np.ndarray]], proprio: np.ndarray, instruction: str, noise: torch.Tensor | None) -> np.ndarray (16,12)`. Message/prompt building MUST reuse `entry.EvalClient._build_messages` semantics — import and call it, byte-identical prompt (test asserts equality on a fixture).
- Noise override WITHOUT editing the checkpoint's `modeling_mibot.py`: wrap the generation call — capture the module's sampler entry (`model.<sample method>`), and replace the initial `torch.randn_like(...)` draw by seeding a local `torch.Generator` and monkeypatching `torch.randn_like` for the duration of the call ONLY if no cleaner seam exists. **Investigation step first:** read `checkpoints/Xiaomi-Robotics-1-RoboCasa365/modeling_mibot.py` lines ~1820–1880 and record (in the report, with line numbers) the exact sampling function signature; prefer subclass-and-override of that one method via `type(model)` subclassing or `types.MethodType` patch. The chosen mechanism must leave server-side behavior untouched.
- [ ] **Step 1 (investigation):** document the sampler seam + KV/hook sites (VLM `language_model.layers[i]`, `dit.layers[i]`, `state_embed`) with line numbers — these are Plan-2's capture contract.
- [ ] **Step 2:** implement runner + noise mechanism; pure test: prompt parity with `EvalClient` on a synthetic observation fixture.
- [ ] **Step 3: determinism gate (the deliverable):** same obs + same noise, two fresh processes → bit-identical `(16,12)` actions (save both npz, `np.array_equal` assert). Different noise → different actions. Record PASS/FAIL in `output/mass_variation/determinism_gate.json`; if bit-identity fails at < 1e-6 max-diff, document and proceed; larger → BLOCKED.
- [ ] **Step 4:** commit.

### Task 5: calibration (Phase 0)

**Files:**
- Create: `eval_robocasa365/mass_variation/calibrate.py` (CLI)

- [ ] **Step 1:** CLI: for each cell, sweep mass ∈ {0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.8} kg (density via `mass_to_density` per sampled mesh probe), 10 episodes each (matched seeds), stock server, recorder ON. Output `output/mass_variation/calibration/<cell>.json`: success + lift rate per mass, knee = heaviest mass with lift-rate ≥ 0.5 falling to < 0.5 at the next level (interpolated midpoint), full provenance (sweep, seeds, server seed, git SHA). Write `output/mass_variation/mass_levels.json` {cell: {light, medium=knee, heavy}}.
- [ ] **Step 2:** CoM axis + offset decision per cell: at knee mass, test ±0.015/±0.025 m on the Task-2-chosen axis with the settle gate; pick the largest offset passing the gate. Append to mass_levels.json (`com_offset_m`, `com_axis`).
- [ ] **Step 3:** wandb run `phase0-calibration`; sanity assertions: lift rate at 0.1 kg ≥ baseline−0.2; monotone-ish degradation (no assertion, but print). Commit calibration outputs (force-add jsons).

### Task 6: Phase-1 benchmark + metrics

**Files:**
- Create: `eval_robocasa365/mass_variation/run_phase1.py` (driver), `eval_robocasa365/mass_variation/metrics.py`
- Test: `eval_robocasa365/mass_variation/test_metrics.py`

- [ ] **Step 1:** driver: 2 cells × 5 conditions × 50 matched seeds, sequential per condition (one worker, one server, `MIBOT_SERVER_SEED=7`), recorder ON, resumable (skip existing ep npz). ~4–5 h; drive in bounded foreground loops.
- [ ] **Step 2:** `metrics.py` (pure, TDD): per condition — success rate, grasp rate, lift rate, t_success (steps→s @20 Hz), drop-after-lift rate (grasped→not-grasped with z falling); table + per-cell CSV `output/mass_variation/phase1/metrics_xr1.csv`.
- [ ] **Step 3:** run, wandb `phase1-benchmark` with the full table + per-condition videos off; report headline (success by mass level per cell, CoM deltas). Commit code + CSV.

---

## Self-review notes

- Spec coverage: cells/conditions/matched-seeds (T1), physics knobs + gates (T2), ground truth + liftoff windows (T3), determinism/noise + capture-site contract for Plan 2 (T4), calibration + CoM decision (T5), Phase-1 benchmark + metrics (T6). Certificates/probing/patching are Plans 2–3 by design (spec Non-goals).
- Known judgment calls: two-pass reset for CoM (settle loop runs inside reset — naive post-reset writes would miss settle dynamics); calibration uses XR1 itself (no scripted policy exists on this stack; 10 trials/level is a knee-finder, not a benchmark); mass sweep in absolute kg (sim defaults unphysical).
- Type consistency: `condition_physics` dict keys used by T2 verify script, T3 npz scalars, T5/T6 drivers match; `episode_seeds` formula identical in T1 test and Global Constraints.

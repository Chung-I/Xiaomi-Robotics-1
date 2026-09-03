# Hidden mass/CoM in Xiaomi-Robotics-1 on RoboCasa365 — study design

Successor to the π0.5/RoboLab study (`RoboLab:docs/studies/2026-09-02-mass-com-vla-probing-design.md`
and its results `2026-09-03-results-pi05-probing.md`). Same questions, new stack, chosen because
RoboLab/Isaac sim throughput was the bottleneck and MuJoCo replay here is bit-exact.

## Questions

1. Does XR1 behaviorally degrade under hidden object mass/CoM changes? (Phase 1)
2. Is hidden mass/CoM/wrench information *present* in XR1's activations? (probing, Plan 2+)
3. Is it *used*? (CRN activation patching at the MoT KV interface, Plan 3)

## Stack facts (verified 2026-09-03 by live exploration; file:line in the exploration reports)

- Model: `MiBoTForActionGeneration` = Qwen3-VL-4B (36 text layers, hidden 2560) + 36-layer DiT
  action expert (hidden 1024), Mixture-of-Transformers: DiT layer i attends the VLM layer-i KV
  cache. Flow matching, 5 Euler steps, chunk (16, 60→12 active dims), replan 16 @ 20 Hz
  (open-loop 0.8 s). Plain PyTorch bf16, no compile/quantization — forward hooks work.
  9.5 GB checkpoint `checkpoints/Xiaomi-Robotics-1-RoboCasa365`.
- Observation (ALL the policy sees): 3 cameras (agentview L/R + wrist) × 4-frame history
  (stride 2, 256², 0.95 crop), proprio 14-d = EEF pos/rot rel base + gripper qpos (2) + base
  pose, prompt with task instruction. NO force/torque, NO joint angles, NO depth.
- RNG: global persistent stream seeded at import (`MIBOT_SERVER_SEED`, default 7+gpu). No
  per-call noise argument (added by this study for offline work).
- Sim: robocasa365 (`~/Codes/robocasa` main @ 921c9a5, editable), mujoco 3.3.1,
  robosuite 1.5.2 source checkout, PandaOmron + PandaGripper, 20 Hz control / 500 Hz physics.
  Bit-exact action replay (verified: 0.0 drift). ~53 steps/s with 3×256² EGL cameras.
- Wrist F/T ground truth: real site sensors `gripper0_right_force_ee/torque_ee`
  (`robot.ee_force/ee_torque`); object wrench `cfrc_ext[bid]` (verified = mg); grasp
  detection `OU.check_obj_grasped`; object pose in obs and `body_xpos/xquat`.
- Mass knob: `density` kwarg at object sampling (uniform 100 default for ALL categories —
  ~10× lighter than physical; 10× density = exactly 10× mass, same CoM/mesh, consistent
  inertia). CoM knob: post-reset `body_ipos` + `body_inertia` writes, BEFORE the settle loop
  (`kitchen.py:1146`), body IDs re-resolved per episode (`hard_reset=True` rebuilds XML).
- Matched pairs: same `episode_seed` + per-episode reseed → identical mesh/scene/placement
  across conditions (verified); `env.rng` is stateful across resets otherwise.

## Design

**Cells (2):** primary `PickPlaceCounterToCabinet` × `milk` (baseline 43/50, sealed opaque
carton). Secondary: `PickPlaceToasterToCounter` × a bottled/boxed category chosen by the
preflight validator (fixture property filters can empty pools; `PickPlaceDrawerToCounter`
excluded — hardcodes its object groups).

**Conditions (5/cell, same shape as before):** MassLight, MassMedium, MassHeavy at CoM-center;
CoMOffA, CoMOffB at MassMedium. Mass levels = {0.3, 1.0, 1.7} × knee, knee calibrated in
Phase 0 with XR1 itself (10 trials/density, sweep over realistic range ~0.1–2.0 kg — the
sim default is unphysically light so the sweep is in absolute kg via density conversion,
measured per mesh). CoM offsets along the object axis where the offset is settle-invisible
(pose gate: post-settle pose delta vs center < 1 mm / 0.5°, ported gate).

**Episodes:** 5 conditions × 2 cells × 50 seeds = 500; per-task horizons (750/450 steps);
identical seed list across conditions (matched by construction).

**Ground truth per step (sidecar npz per episode):** ee_force/ee_torque (6), cfrc_ext of the
target object (6), object pose (7), gripper qpos (2), grasp flag, actions applied (12),
EEF pose (proprio copy), lift-off step (first z-rise ≥ 0.05 m while grasped).

**Windows (lesson: design around the physics):** phase masks are defined from grasp/lift-off
events at analysis time — precontact (before first grasp contact), grasp-to-liftoff, CARRY
(airborne, the primary mass window), post-place. No proxy anchors.

**Certificates (pre-registered, lesson from Amendment 4 discussion):**
- Raw-physics certificate: ridge/GRU on wrist F/T (+proprio) → is mass in the sim signals?
- POLICY-OBSERVABLE certificate: same models on exactly XR1's inputs — 14-d proprio history
  (4×stride-2, as the policy sees) with and without downsampled wrist-camera frames. This is
  the decisive control: a probe null is a *representation* claim only where this passes; where
  it fails, the claim is an *input-availability* (sensing-gap) finding about the embodiment.
- Untrained-copy (random-init) bound, ported.

**Determinism policy:** Phase 0/1 use the stock socket server with `MIBOT_SERVER_SEED=7`
pinned and single-worker-per-run mapping recorded (RNG-stream nondeterminism accepted, as in
the π0.5 study). All capture/probing/patching runs use the in-process runner with an explicit
`noise` argument (added without editing the checkpoint: subclass/patch at load time) and a
bit-exactness gate.

**Patching interface (Plan 3, recorded now):** the MoT coupling makes the VLM per-layer KV
cache the clean causal boundary — patch layer-i KV (per token block: per-camera image tokens,
text, state token) fed to the DiT; DiT-side patches re-applied at each of the 5 flow steps.
CRN via the added noise argument; both directions; reseed + degradation floors; frozen pairs.

**Metrics discipline (ported from amendments 1–3):** within-object primary mass target
(`mass_log_c` relative to per-cell knee); rank_acc + RMSE secondaries; degenerate guard;
selectivity with group-coherent shuffles and per-draw alpha search; episode-grouped CV;
identity channel reported alongside (here: object-mesh identity within category).

## Non-goals now

Plans 2–3 documents are written only after Phase 0/1 artifacts exist (sequencing that worked).
No robocasa/robosuite edits (monkeypatch from study code); no upstream pushes (origin =
XiaomiRobotics — a fork will be added as push remote).

## Repos & infra

Branch `study/mass-com-xr1` in `~/Codes/Xiaomi-Robotics-1`; study code under
`eval_robocasa365/mass_variation/`; robocasa venv (`~/Codes/robocasa/.venv`) drives sim,
`.venv-mibot` drives the model server; `MUJOCO_GL=egl`; wandb project `mass-com-xr1`
(mandatory for all runs); local RTX 5090.

## Addendum (2026-09-03, user directives — binding)

1. **Headline channel: load-dependent EEF tracking error.** The named question of this study
   is whether the model can extract mass/tactile information from the discrepancy between
   commanded EE deltas (its own actions) and achieved EEF motion (its proprio history) under
   load. Pre-registered accordingly: (a) a dedicated tracking-error certificate — inputs =
   commanded-vs-achieved EEF delta history exactly as reconstructable from the policy's own
   observation window (4 frames, stride 2 for XR1; single frame for π0.5); (b) the
   cross-model prediction, registered before any run: XR1's 4-frame history contains this
   channel, π0.5's single-frame observation largely does not — if tracking error is the mass
   channel, XR1 can encode hidden mass where π0.5 architecturally cannot. Both directions of
   outcome are informative.
2. **Second model: π0.5 (robocasa365-trained).** Official checkpoint
   `robocasa/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000` (HF),
   served via the `robocasa-benchmark/openpi` fork (commit ca4c6d7, Atomic-Seen 39.6%).
   Same cell, same fixed mass levels, same matched seed list. Risk recorded: π0.5's per-task
   baseline on the primary cell may be low (~0.3–0.5); if its baseline success is < 0.25 in
   the sanity sweep, switch the primary cell to the highest-joint-baseline PickPlace task
   before Phase 1 (decision rule, not post-hoc).
3. **Budget: ≤ 1 h per model.** Cuts, in order applied: CoM arm DEFERRED from Phase 1
   (conditions 5→3: MassLight/Medium/Heavy); secondary cell DEFERRED (cells 2→1); behavioral
   knee calibration REPLACED by fixed physical mass levels **{0.15, 0.6, 1.2} kg**
   (identical for both models; a 3-level × 5-trial sanity sweep per model replaces Phase 0);
   seeds 50→35. Phase 1 = 3 × 1 × 35 = 105 episodes ≈ 1 h per model. Deferred arms remain
   specified above and can be run later without design changes.

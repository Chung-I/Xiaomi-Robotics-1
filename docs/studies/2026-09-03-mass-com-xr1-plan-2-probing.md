# XR1 Mass/CoM Plan 2: Certificates + XR1 Probing (certificate-gated π0.5 branch)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, with certificates, whether each model's own observation carries hidden-mass information; probe XR1's activations for it (XR1 behaviorally adapts — flat success over 8× mass — so its internals owe an explanation); gate any π0.5 capture on its certificate.

**Architecture:** Pure analysis (certificates, probes) runs on the 210-episode Phase-1 npz corpus plus a one-shot replay extraction of policy-format states; XR1 activation capture couples the bit-exact replay loop (robocasa venv) to a hooked in-process model behind a local capture server (.venv-mibot) — two venvs, one socket, no frame dump.

**Tech Stack:** robocasa venv (`~/Codes/robocasa/.venv`) for sim/replay + sklearn/pandas analysis (install ad-hoc: `uv pip install scikit-learn` with that venv, record versions); `.venv-mibot` for the capture server (torch, XR1Runner). wandb project `mass-com-xr1`.

**Spec:** `docs/studies/2026-09-03-mass-com-xr1-design.md` + its Addendum, Correction (achieved-only certificates), and Correction 2. Methods discipline ported from the π0.5/RoboLab study's plan-3 (amendments 1–3 there): R²+selectivity primary, rank_acc secondary, degenerate guard, episode-grouped CV, group-coherent shuffles with per-draw alpha search.

## Global Constraints

- Data: `output/mass_variation/phase1/PickPlaceCounterToCabinet/<cond>/ep_<seed>.npz` (XR1) and `...__pi05_robocasa/<cond>/ep_<seed>.npz` (π0.5); 35 seeds (7..41) × 3 conditions × 2 models; per-step channels incl. `commanded_delta (T,6)` base-frame, `achieved_eef_delta (T,6)` world-frame (conjugation caveat documented in recorder.py), `ee_force/ee_torque`, `cfrc_obj` ([torque,force], +mg), `grasped`, `obj_pos`, scalars incl. `mass_kg` (measured), `seed`, `success`, `liftoff_step`.
- **Phase masks (from events, never proxies):** `precontact` = before first grasped step; `grasp` = first-grasp..liftoff; `carry` = airborne (liftoff_step ≤ t < first step after liftoff where obj z < init+0.02 or episode end); `all`. Mask builders are pure and unit-tested.
- **Primary mass target:** `mass_log_c = log(mass_kg / 0.6)` — support {log 0.25, 0, log 2} identical across conditions by construction. Secondary metrics per the ported discipline (rank_acc from held-out predictions, chance 0.5; RMSE for force targets). Degenerate guard: masked target variance < 1e-12 → NaN + flag.
- **CV:** grouped by episode (seed), GroupKFold(5), 35 groups per model; selectivity = real − mean of 5 group-coherent shuffles, each with its own alpha search (alphas 10^-2..10^4).
- **Pre-registered certificate gates (before any result is seen):** mass recoverable at ridge R² ≥ 0.3 on `carry` for a certificate to PASS; the sequential rule: XR1-format certificate PASS + XR1 probe null → certified null; certificate FAIL → uncertified (data/sensing limitation), stated as such. π0.5 branch rule in Task 5.
- **Registered expectations (written now):** XR1-format (4×14-D achieved-only history) certificate plausibly PASSES on carry (sag/velocity signatures); π0.5-format (16-D single frame) certificate plausibly FAILS (no motion in a single frame — a static pose carries mass only via residual sag); K_eff regression (in-carry wrist F_z vs per-step commanded−achieved z-deficit) expected strongly linear. Both directions of every outcome are reportable.
- Analysis code lives in `eval_robocasa365/mass_variation/analysis/`; the probe core is COPIED from the sibling study (`~/Codes/RoboLab/.claude/worktrees/mass-com-vla-probing/analysis/mass_com/probe_core.py`, our own code) with an origin note in the header — do not reimplement it, do not import across repos.
- Ops: never push (controller pushes to `mine`); one model process on the GPU at a time; fresh server per batch where a server is used; bracket-trick ps, exact-PID kills; suite `~/Codes/robocasa/.venv/bin/python -m pytest eval_robocasa365/mass_variation -v` stays green (86 + new); every run logs to wandb `mass-com-xr1` with full config.

---

### Task 1: policy-state replay extraction + analysis dataset

**Files:**
- Create: `eval_robocasa365/mass_variation/extract_policy_state.py` (CLI), `eval_robocasa365/mass_variation/analysis/__init__.py`, `eval_robocasa365/mass_variation/analysis/dataset.py`
- Test: `eval_robocasa365/mass_variation/test_analysis_dataset.py`

**Interfaces:**
- Produces: per episode `output/mass_variation/policy_state/<model>/<cond>/ep_<seed>.npz` with `obs_state_14 (T,14)` (XR1 field order: EEF pos rel base, EEF rot axis-angle, gripper qpos, base pos, base rot — exactly `entry.py:58-71` semantics via the same helpers) and `obs_state_16 (T,16)` (π0.5 field order per `openpi-robocasa main.py:132-142`, raw quat) — BOTH extracted in ONE replay pass per episode by reading the gym observation dict each step (the wrapper exposes every needed key; reuse `render_episode.py`'s bit-exact replay scaffold, rendering OFF, and its intended-mass density derivation + measured-scalar assert).
- `analysis/dataset.py`: `load_study(model: str, phase1_root, policy_state_root) -> dict` — concatenated per-step arrays over 105 episodes with `episode_id, seed, condition, step, mass_kg, mass_log_c, masks {precontact, grasp, carry, all}, deficit_z (T,)` (= commanded_delta[:,2] − achieved_eef_delta[:,2] per step), plus passthrough of force/tracking channels and `policy_state_14/16`; `window_stack(x, k=4, stride=2)` (left-edge clamp, pure) for building the XR1 4-frame proprio window at analysis time.

- [ ] **Step 1 (TDD, pure):** mask builders (`phase_masks(grasped, liftoff_step, obj_z, init_z)` — hand-built fixtures: no-lift episode → carry empty; drop mid-carry ends carry at the z-return step; boundary exact), `mass_log_c` exactness (0.15→log 0.25), `window_stack` shapes/clamping, `deficit_z` sign convention (commanded down minus achieved down). RED → implement → GREEN.
- [ ] **Step 2:** `extract_policy_state.py`: replay each episode (both models' corpora), assert bit-exactness (success + liftoff_step equal the npz scalars — abort the episode on divergence, report it, never loosen), write the state npz. Batch over all 210; resumable; ~15 s/episode CPU → ~55 min; drive in bounded foreground loops. Spot-verify on 3 episodes: `obs_state_14`'s EEF-pos channel differs from the world-frame `eef_pos` npz channel exactly by the base transform (sanity print).
- [ ] **Step 3:** run the full extraction; commit code + a small `policy_state_manifest.json` (force-add; npz untracked).

### Task 2: probe core port + certificates (the decisive experiment)

**Files:**
- Create: `eval_robocasa365/mass_variation/analysis/probe_core.py` (copied, header note), `eval_robocasa365/mass_variation/analysis/certificates.py` (CLI)
- Test: `eval_robocasa365/mass_variation/test_probe_core_port.py` (smoke: the 4 behavioral tests from the origin suite, adapted paths)

**Interfaces:**
- Consumes: Task 1 dataset loader.
- Produces: `output/mass_variation/analysis/certificates.json` — per (model, certificate, mask): pooled R², per-fold R²s, rank_acc, gate PASS/FAIL, input-channel list. Certificates, all ridge (+ small GRU for the two policy-observable ones, torch on GPU allowed — one process, ≤5 min/target, seeded):
  1. `raw_ft`: wrist ee_force+ee_torque windows (k=16) → mass_log_c. Physics channel.
  2. `policy_obs_xr1`: `window_stack(obs_state_14)` (4×14, stride 2 — exactly the policy's format, achieved-only) → mass_log_c. THE headline certificate.
  3. `policy_obs_pi05`: `obs_state_16` single frame → mass_log_c. The sensing-gap test.
  4. `deficit`: commanded/achieved delta windows (k=8) → mass_log_c. Analysis-side channel content.
  5. `k_eff`: in-carry regression F_z ~ deficit_z per step (per model): report slope (K_eff), R², per-condition means. Physical validation, not gated.
- No-circularity rule: certificates 2–3 use ONLY policy-observable channels (no forces); certificate 1 may not use policy states.

- [ ] **Step 1:** copy probe_core.py verbatim + origin header; adapt imports; run the 4 ported behavioral tests → GREEN (they encode grouped-CV + selectivity semantics).
- [ ] **Step 2:** certificates CLI with the 5 certificates; run on BOTH models' datasets; wandb run `plan2-certificates`; print the gate table.
- [ ] **Step 3:** commit code + certificates.json (force-add). The controller reads the gate table before Tasks 3/5 proceed — report it prominently.

### Task 3: XR1 activation capture (replay ↔ capture server)

**Files:**
- Create: `eval_robocasa365/mass_variation/capture_server.py` (.venv-mibot), `eval_robocasa365/mass_variation/replay_capture.py` (robocasa venv)
- Test: `eval_robocasa365/mass_variation/test_capture_protocol.py` (pure: message pack/unpack, window assembly)

**Interfaces:**
- Protocol: length-prefixed pickle over localhost (mirror `deploy/server.py`'s framing): request {episode_id, step, frames_by_cam (4×256²×3 uint8 ×3 cams), proprio_history (4,14), instruction, noise_seed} → response {ok}; server writes acts to `output/mass_variation/activations/xr1/<cond>/ep_<seed>.npz` incrementally.
- Capture grid (per the T4 capture contract in the Plan-1 history — hook sites: `model.vlm.model.language_model.layers[i]` i=0..35, `model.dit.layers[j]` j=0..35 × 5 flow steps, `state_embed`): capture at REPLAN steps only (every 16 recorded steps — the policy's actual decision points, ~44/episode); positions per prefix token blocks {last_prefix_token, image_tokens_mean (per camera), state_token} + DiT taps at flow steps {0,4}; fixed noise per (episode, step) from `default_rng(noise_seed)`; f16; meta.json with layer names, positions, token-block ranges, git SHA, determinism note.
- Budget: ~4.6k inferences ≈ 30–45 min GPU; acts ≈ 4.6k × (36×5 + 36×2×...) — keep ≤ 4 GB total (compute exact in-code and abort if projected over 8 GB).

- [ ] **Step 1 (pure TDD):** protocol pack/unpack round-trip; window assembly from a synthetic frame stream (indices t−6,t−4,t−2,t with left clamp — must match `entry.py:sample_history` exactly, import it for the reference).
- [ ] **Step 2:** capture server (XR1Runner + hooks + fire-count asserts per the T4 contract; `TORCHDYNAMO` not used here but keep the runner's noise mechanism); determinism gate: one (episode, step) captured twice across two server restarts → bit-identical acts + actions.
- [ ] **Step 3:** replay_capture over all 105 XR1 episodes (server on 10087; fresh start; exact-PID lifecycle); resumable; drive bounded. Commit code + meta.json (acts npz untracked); wandb `plan2-capture-xr1`.

### Task 4: XR1 probing + random-init bound

**Files:**
- Create: `eval_robocasa365/mass_variation/analysis/run_probes_xr1.py` (CLI)
- Test: covered by ported core; run-level sanity asserts inline.

- [ ] **Step 1:** join acts (replan steps) with labels/masks at those steps; grid: targets {mass_log_c, wrench_norm (√(fx²+fy²+fz²) of ee_force), deficit_z, contact flag (grasped), step_clock control, object-mesh identity control (clf over mesh ids from ep_meta if recorded; else skip and note)} × VLM layers 0..35 × positions × masks {precontact, carry, all}; sanity gates: step_clock decodes (ceiling analog) somewhere > 0.9; mass_log_c precontact selectivity < 0.1 everywhere (leakage guard — abort loudly).
- [ ] **Step 2:** random-init bound: capture pass with `XR1Runner` random-init weights (Plan-1 runner supports fresh-init load? if not, add `random_init=True` mirroring the sibling study's approach — seeded, no checkpoint load) at key layers {0, 11, 23, 35} over 1 condition-triple subset (~15 episodes), probe the same cells.
- [ ] **Step 3:** results parquet + figures (R² vs layer per mask; carry-mask headline) + wandb `plan2-probes-xr1`; commit code + parquet summary (force-add small files).

### Task 5: π0.5 branch (certificate-gated — pre-registered rule)

- Rule (binding, no third path): if `policy_obs_pi05` certificate FAILS its gate → NO π0.5 capture; the conclusion is a SENSING-GAP finding ("π0.5's single-frame observation does not carry the hidden-mass channel; its behavioral mass-sensitivity, if real, must route through vision or is noise") and Task 5 is a one-paragraph section in the results doc. If it PASSES → STOP and report to the controller (capture would require converting the robocasa π0.5 checkpoint to PyTorch — a separate scope the controller must authorize).
- [ ] **Step 1:** implement nothing until Task 2's gate table exists; then execute the branch per the rule; ledger the outcome.

### Task 6: results doc + deck update

**Files:**
- Create: `docs/studies/2026-09-04-results-xr1-mass-probing.md`
- Modify: `~/Codes/daily-logs/researches/property-belief-manipulation/slides/advisor-1pager.tex` (update the Study-2 line in the hypothesis block with the outcome, one clause; rebuild + render-check; commit+push daily-logs per its flow)

- [ ] **Step 1:** results doc: behavioral recap (both models), certificate table with gates, K_eff physical validation, XR1 probe results read strictly through the sequential rule, π0.5 branch outcome, controls, limitations (n=35 episodes/model, one cell, sim-only, replan-step capture granularity), wandb links, provenance footer. Every number scripted-verified against its artifact (verification output committed to the SDD workspace).
- [ ] **Step 2:** deck: replace "Runs start next"-era phrasing with the measured outcome (one line, no overflow — render check). Commit both repos (push daily-logs only; the controller pushes the study repo).

---

## Self-review notes

- Spec coverage: certificates incl. corrected achieved-only scoping + K_eff promise (design Correction ¶) → T2; carry-anchored masks → T1; XR1 probing of the adaptation finding → T3/T4; π0.5 single-frame question → T2 cert 3 + T5 rule; deck follow-through → T6. CoM arm remains deferred (Plan-1 scope), patching (MoT KV) is Plan 3 — out of scope here by design.
- Judgment calls recorded: capture at replan steps only (decision-point fidelity over density; ~44 samples/episode × 105 = ~4.6k rows is enough for grouped CV at 35 groups); two-venv capture solved by a local server rather than frame dumps; probe core copied not imported (repo isolation).
- Type consistency: `load_study` keys consumed by T2 CLI and T4 join; `window_stack(x, k=4, stride=2)` used in cert 2 and T3 window assembly reference; certificate names in T2 match T5's rule and T6's table.

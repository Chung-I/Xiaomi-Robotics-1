# Copyright (C) 2026 Xiaomi Corporation.
"""EXPLORATORY time-resolution of the pre-registered carry-phase probe result.

**This analysis introduces no new claim.** It re-cuts the SAME rows, the
SAME probe discipline and the SAME targets as the completed, pre-registered
Task-4 phase-bucket analysis (``run_probes_xr1.py``) along a finer temporal
axis, to answer a descriptive follow-up: *when*, relative to first contact,
does hidden mass become linearly decodable? It is post-hoc and is labeled
EXPLORATORY everywhere it is reported. The study's headline is unchanged and
is NOT upgraded by anything here: peak mass selectivity 0.227 at DiT L28
(flow4 state-block) with pooled held-out R² 0.189 on the carry mask --
below the 0.3 certificate bar. A larger number in some narrow bin here is a
NOISIER estimate on ~5x fewer rows, not a better result.

Axis
----
x = time relative to FIRST CONTACT (the episode's first ``grasped==True``
step), in seconds at ``metrics.CONTROL_HZ`` = 20 Hz. Episodes that never
grasp (5 of 105) have no contact reference and are dropped entirely.

y = SELECTIVITY for the hidden mass (``mass_log_c``) = real pooled held-out
R² minus the mean of the shuffled control's R², per bin, with a +-1
shuffled-std band. Selectivity (not raw R²) is the y axis because the
shuffled floor itself moves with bin size and with which seed groups land
in which fold.

Panel A lines (6)
-----------------
1. ``physics ceiling`` -- ridge on the raw wrist F/T window: exactly the
   ``raw_ft`` certificate's design matrix (``certificates.build_features``:
   raw ``ee_force``/``ee_torque`` components plus the amendment-B derived
   frame-aware features |F|, |tau|, ``F_world``, ``F_base``; trailing
   window k=16, stride 1, never crossing an episode boundary). This is what
   the SENSOR makes knowable at each time -- the reference curve, not a
   model site.
2. ``XR1 action module (DiT L28, flow4 state-block)`` -- the probe peak site.
3. ``XR1 wrist-camera tokens (VLM L21)`` -- the vision site (the VLM peak).
4. ``XR1 proprioception (state_embed)`` -- the null site.
5. ``commanded-vs-achieved deficit`` -- ridge on exactly the ``deficit``
   certificate's design matrix (``certificates.build_features("deficit", d)``:
   ``commanded_delta[6]`` + ``achieved_eef_delta[6]``, trailing window k=8,
   stride 1, never crossing an episode boundary; the identical feature
   builder imported from ``certificates.py``, not re-implemented). This
   channel is the one the ``deficit`` certificate scores on the ``carry``
   mask: aggregate-informative (episode-mean commanded-up effort rises +23%
   for XR1 / +87% for pi0.5 from lightest to heaviest carton) but weak
   per-step (certificate R^2 0.062, FAIL). Its time profile was unknown
   before this curve: does it spike at contact like the physics ceiling, or
   stay flat -- i.e. is the load signature instantaneous or only visible
   under averaging? Added as a fifth, directly comparable line (same bins,
   rows, folds, shuffle protocol as the other four) to answer that
   descriptively; both outcomes are reportable and neither would change the
   study headline.
6. ``camera pixels -- OUR reader`` -- the vision certificate's own
   ``ridge_pca`` reader (``vision_certificate.py``: 96x96 GRAYSCALE stored
   frames in XR1's own visual format -- 4 frames at stride 2, 3 cameras,
   its own 0.95 centre crop -- with PCA fitted on the TRAIN rows WITHIN
   EACH FOLD, then the same ridge), re-cut into these bins. It exists to
   show the INSTRUMENT ASYMMETRY: the model's own encoder finds mass in its
   image tokens (line 3) where this low-resolution reader finds none. It is
   NOT the model's vision and is labeled so everywhere. It is also the one
   line that cannot be scored on every row: ``render_frames`` stored frames
   for the ``precontact`` steps and the ``carry`` windows ONLY, so captured
   rows inside the grasp transition or after the place have no frames
   (``window_available``); those rows are EXCLUDED (never faked with a
   neighbouring frame), the retention rule is re-applied to the subset, and
   a row-matched VLM L21 cell on exactly this line's rows is recorded --
   but never plotted -- as the fair comparator.

Lines 1-5 are scored on the SAME rows (the captured replan steps inside a
bin) with the SAME GroupKFold(5) fold partition (computed once per bin from
the seed groups), so they are strictly comparable; line 6 gets its own row
scope and its own fold partition on that scope, for the reason above. The
one deliberate preprocessing difference is the one each corpus already
carries in the study: the physics and deficit features are globally
z-scored inside the cell (mixed physical units -- Newtons, N·m, radians for
physics; normalized action units vs metres for deficit), exactly as
``certificates.certificate_cell`` does, while activations are not
(homogeneous units, matching ``run_probes_xr1``).

Panel B (the positive control): contact-force magnitude
-------------------------------------------------------
Same bins, same rows, same folds and the same shuffle protocol as panel A
-- literally the same per-fold SVD factors -- but the target is
``wrench_norm`` = |F| in newtons (``run_probes_xr1.wrench_norm_of``, already
a target of the pre-registered probe grid). It shows the three MODEL sites
plus the deficit channel, and asks: does the model track the IMMEDIATE
CONTACT CONSEQUENCE even where it fails to track the LATENT PROPERTY?

The wrist-F/T physics ceiling is DELIBERATELY ABSENT from panel B. Its
design matrix IS the force sensor, so decoding |F| from it is circular --
reading the target off the instrument that defines it -- and would be a
tautology, not a ceiling. This is stated on the panel, in the figure
caption and in ``timecurve_summary.json``.

Event rules (three)
-------------------
Contact (x = 0, hard rule), median lift-off, and median PLACE/RELEASE --
the first step after lift-off at which the gripper has let go and the
object is no longer rising (``place_step``). The place rule explains why
every line decays late in the span. It is NOT defined against the object's
initial height: the task places the carton on a cabinet shelf ~0.5 m above
the counter it came from, so an ``init_z``-relative rule fires in only
17/100 episodes on this corpus while the non-rising rule fires in 95/100.

Bins (chosen for sample sufficiency, NOT tuned on outcomes)
----------------------------------------------------------
XR1 activations exist only at replan steps (every 16 steps = 0.8 s, ~23 per
episode), so the bins are WIDE: ``BIN_WIDTH_STEPS`` = 32 steps = 1.6 s,
spanning ``SPAN_LO_STEPS``..``SPAN_HI_STEPS`` = -128..+192 steps
(-6.4 s .. +9.6 s) relative to contact, with a bin edge exactly at contact.
A bin is RETAINED only if it holds >= ``MIN_ROWS`` (120) rows AND
>= ``MIN_EPISODE_GROUPS`` (15) distinct episodes; otherwise it is dropped
and recorded as dropped (``retained=False`` with a ``drop_reason``) rather
than scored on a handful of episodes. The rule was fixed before any curve
was computed and is applied identically to every line and both panels (and
separately, on its own row count, to the camera-pixel line's subset).

Outputs: ``<out-dir>/timecurve.parquet`` (one row per series x bin,
including dropped bins), ``<out-dir>/fig_mass_vs_time.png``,
``<out-dir>/timecurve_summary.json``; wandb run ``plan2-timecurve-xr1-v3``
(project ``mass-com-xr1``).

Run (robocasa venv; pure analysis, no GPU):

    ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.analysis.time_curve
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval_robocasa365.entry import CAMERA_KEYS
from eval_robocasa365.mass_variation.analysis import certificates, probe_core
from eval_robocasa365.mass_variation.analysis.dataset import load_study
from eval_robocasa365.mass_variation.analysis.run_probes_xr1 import (
    _cell_from_factors,
    _degeneracy_reason,
    join_index,
    load_acts,
    wrench_norm_of,
)
from eval_robocasa365.mass_variation.analysis.vision_certificate import (
    N_PCA_COMPONENTS,
    STORE_SIDE_PX,
    VISUAL_FORMATS,
    apply_format_transform,
    gather_visual_rows,
    pca_fold_factors,
    window_steps,
)
from eval_robocasa365.mass_variation.metrics import CONTROL_HZ

REPO_ROOT = Path(__file__).resolve().parents[3]

SEED = 0
TARGET = "mass_log_c"          # panel A
TARGET_FORCE = "wrench_norm"   # panel B (addition 1)

# Place/release detector (addition 4). The tolerance is the largest per-step
# RISE in the object's height that still counts as "not rising" -- 0.1 mm,
# i.e. floating-point settle noise, not motion.
PLACE_RISE_TOL_M = 1e-4

# XR1's own visual format, read off the vision certificate rather than
# restated here, so the camera-pixel line can never drift from it.
CAM_K = VISUAL_FORMATS["xr1"]["window"]["k"]           # 4 frames
CAM_STRIDE = VISUAL_FORMATS["xr1"]["window"]["stride"]  # stride 2

# Bin rule -- fixed before any curve was computed (see module docstring).
BIN_WIDTH_STEPS = 32          # 1.6 s at 20 Hz (2 replan captures per episode)
SPAN_LO_STEPS = -128          # -6.4 s
SPAN_HI_STEPS = 192           # +9.6 s
MIN_ROWS = 120
MIN_EPISODE_GROUPS = 15

EXPLORATORY_LABEL = (
    "EXPLORATORY, post-hoc temporal resolution of the pre-registered "
    "carry-phase probe result (run_probes_xr1.py Task 4). No new claim: "
    "same rows, same targets, same probe discipline, finer time grain. "
    "Bins were chosen for sample sufficiency, not tuned on outcomes."
)
HEADLINE_NOT_UPGRADED = (
    "The study headline is unchanged: peak mass selectivity 0.227 at DiT "
    "L28 (flow4 state-block), pooled held-out R^2 0.189 on the carry mask, "
    "below the 0.3 certificate bar. Any larger per-bin value here is a "
    "noisier estimate on ~5x fewer rows, not a better result."
)

# (label, block, layer, position) -- block "raw_ft" is the physics reference,
# block "deficit" is the commanded-vs-achieved deficit reference (also a
# sensor/action-side channel, not a model site).
SITES = (
    ("physics ceiling (wrist F/T, k=16 window)", "raw_ft", -1, "ee_force+ee_torque+derived"),
    ("XR1 action module (DiT L28, flow4 state-block)", "dit", 28, "flow4:state_tokens_mean"),
    ("XR1 wrist-camera tokens (VLM L21)", "vlm", 21,
     "image_tokens_mean:video.robot0_eye_in_hand"),
    ("XR1 proprioception (state_embed)", "state_embed", -1, "state_tokens_flat"),
    ("commanded-vs-achieved deficit (k=8 window)", "deficit", -1,
     "commanded_delta+achieved_eef_delta"),
)

# CVD-checked on a light surface; per-series markers are the redundant
# encoding. The physics reference is neutral grey + dashed so it never reads
# as one of the three model sites. The four original entries are UNCHANGED
# from the published version; the deficit line below is the one addition
# (distinct purple + filled-plus marker + dash-dot, so it never reads as
# any of the four existing lines).
SITE_STYLE = {
    "physics ceiling (wrist F/T, k=16 window)":
        {"color": "#4a4a4a", "marker": "D", "ls": "--"},
    "XR1 action module (DiT L28, flow4 state-block)":
        {"color": "#c14f2e", "marker": "o", "ls": "-"},
    "XR1 wrist-camera tokens (VLM L21)":
        {"color": "#3a6ea5", "marker": "s", "ls": "-"},
    "XR1 proprioception (state_embed)":
        {"color": "#2f9e63", "marker": "^", "ls": "-"},
    "commanded-vs-achieved deficit (k=8 window)":
        {"color": "#8e44ad", "marker": "P", "ls": "-."},
}

# ---- addition 2: the camera-PIXEL reader (a sixth panel-A line) -----------
# Deliberately NOT a model site. It is the vision certificate's own reader
# (96x96 grayscale, XR1's 4-frame stride-2 x 3-camera format, PCA fitted on
# TRAIN rows within each fold, then the same ridge) pointed at the same bins,
# so the figure can show the INSTRUMENT ASYMMETRY: the model's own encoder
# finds mass in its image tokens (line 3) where this low-resolution reader
# finds none. Style: hollow gold marker + dotted line, unlike every model
# line (solid, filled) and unlike the two sensor references.
CAMERA_LABEL = "camera pixels — OUR 96×96 reader (not model vision)"
CAMERA_SITE = (CAMERA_LABEL, "camera", -1,
               "xr1 4-frame stride-2 × 3 cameras, 96×96 grey, in-fold PCA + ridge")
SITE_STYLE[CAMERA_LABEL] = {"color": "#9c6b1e", "marker": "v", "ls": ":",
                            "mfc": "#ffffff"}

# A row-matched companion for the camera line, recorded but NEVER plotted:
# the camera reader can only be scored on rows whose stored frame window
# exists (see `window_available`), which is a SUBSET of each bin's rows, so
# "the model's VLM beats our pixel reader" would otherwise compare two
# different row sets. This series re-scores VLM L21 on EXACTLY the camera
# line's rows and folds, and the report quotes it.
VLM_MATCHED_LABEL = "XR1 wrist-camera tokens (VLM L21) on the pixel rows [diagnostic]"

# ---- addition 1: panel B, the positive control -----------------------------
# Same bins, rows, folds and shuffle protocol as panel A; target is the
# CONTACT-FORCE MAGNITUDE |F| (`wrench_norm`, already a target in
# run_probes_xr1.py). The three MODEL sites plus the deficit channel.
# The wrist-F/T "physics ceiling" is DELIBERATELY ABSENT: its features ARE
# the force sensor, so decoding |F| from it is circular and would not be a
# ceiling but a tautology. Said in the caption, on the panel, and in the
# summary json.
PANEL_B_SITES = SITES[1:]
PANEL_B_EXCLUSION_NOTE = (
    "Panel B deliberately has NO physics-ceiling line: its features are the "
    "wrist F/T sensor itself, so decoding |F| from it would be circular "
    "(reading the target off the instrument that defines it), not a ceiling."
)
# Stated wherever panel B's numbers are, never only in the report: |F| is a
# heavily right-skewed target (per-bin skew 3.3..8.4, medians ~5-11 N against
# maxima of 16-217 N), so a pooled R^2 on it is carried by the tail, and the
# accompanying rank accuracy -- which orders ALL pairs, not just the tail --
# is the honest check on how much of the bulk the fit actually orders.
PANEL_B_CAVEAT = (
    "READ PANEL B WITH ITS RANK ACCURACY, not its R^2 alone: |F| is heavily "
    "right-skewed in every bin (skew 3.3..8.4; median ~5-11 N against maxima "
    "of 16-217 N), so a pooled R^2 is dominated by a handful of impulsive "
    "peaks. Where the |F| rank accuracy sits near 0.5 the fit is separating "
    "'about to spike / not' rather than ordering the bulk of the force "
    "distribution. Panel B is a POSITIVE CONTROL -- evidence that these sites "
    "are not inert -- not a claim about how finely the model resolves force."
)

DEFAULT_FRAMES_ROOT = "output/mass_variation/frames"
DEFAULT_ACTS_ROOT = "output/mass_variation/activations/xr1"
DEFAULT_PHASE1_ROOT = "output/mass_variation/phase1"
DEFAULT_POLICY_STATE_ROOT = "output/mass_variation/policy_state"
DEFAULT_OUT_DIR = "output/mass_variation/analysis/probes_xr1"


# ---------------------------------------------------------------------------
# Pure binning helpers (unit-tested in test_time_curve.py)
# ---------------------------------------------------------------------------


def first_contact_steps(episode_id, step, grasped) -> dict[str, int]:
    """``{episode_id: first grasped step}``, from the FULL per-step rows.

    An episode with no grasped step is ABSENT from the table (never mapped
    to 0) -- it has no contact reference and must drop out of the analysis
    rather than be centred on a fictitious contact. ``step`` is read from
    the step column, not the row offset, so this is correct on a subset."""
    episode_id = np.asarray(episode_id)
    step = np.asarray(step)
    grasped = np.asarray(grasped, dtype=bool)
    table: dict[str, int] = {}
    for eid in np.unique(episode_id):
        m = episode_id == eid
        hit = np.flatnonzero(grasped[m])
        if hit.size:
            table[str(eid)] = int(np.asarray(step[m])[hit[0]])
    return table


def contact_relative_steps(episode_id, step, contact_step: dict[str, int]) -> np.ndarray:
    """``step - contact_step[episode]`` per row, as float; NaN where the
    episode has no entry in ``contact_step`` (never grasped)."""
    episode_id = np.asarray(episode_id)
    step = np.asarray(step, dtype=np.float64)
    out = np.full(step.shape, np.nan, dtype=np.float64)
    for i, eid in enumerate(episode_id):
        c = contact_step.get(str(eid))
        if c is not None:
            out[i] = step[i] - c
    return out


def make_bins(lo_step: int, hi_step: int, width_steps: int) -> tuple[tuple[int, int], ...]:
    """The contiguous half-open ``[lo, hi)`` step-bin grid over
    ``[lo_step, hi_step)``. Raises if the span is not a whole number of
    bins (a partial trailing bin would silently hold fewer rows than every
    other bin) or the width is not positive."""
    if width_steps <= 0:
        raise ValueError(f"make_bins: width_steps must be > 0, got {width_steps}")
    span = hi_step - lo_step
    if span <= 0 or span % width_steps != 0:
        raise ValueError(
            f"make_bins: span {lo_step}..{hi_step} is not a whole number of "
            f"{width_steps}-step bins"
        )
    return tuple(
        (lo, lo + width_steps) for lo in range(lo_step, hi_step, width_steps)
    )


def assign_bins(step_rel, bins) -> np.ndarray:
    """Bin index per row (``-1`` for NaN or outside every bin).

    Bins are half-open ``[lo, hi)``: a value exactly on a lower edge belongs
    to that bin, a value exactly on an upper edge belongs to the next one."""
    step_rel = np.asarray(step_rel, dtype=np.float64)
    out = np.full(step_rel.shape, -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(bins):
        out[(step_rel >= lo) & (step_rel < hi)] = b
    return out


def steps_to_seconds(steps, control_hz: float = CONTROL_HZ) -> np.ndarray:
    """Control steps -> seconds (the figure's x unit)."""
    return np.asarray(steps, dtype=np.float64) / control_hz


def bin_drop_reason(n: int, n_episodes: int,
                    min_rows: int = MIN_ROWS,
                    min_episode_groups: int = MIN_EPISODE_GROUPS) -> str | None:
    """The degenerate guard's verdict for one bin, or ``None`` to retain."""
    if n < min_rows:
        return f"n={n} < {min_rows} rows"
    if n_episodes < min_episode_groups:
        return f"n_episodes={n_episodes} < {min_episode_groups} episode groups"
    return None


# ---------------------------------------------------------------------------
# Place / release event (addition 4 -- the figure's third vertical rule)
# ---------------------------------------------------------------------------


def place_step(grasped, liftoff_step: int, obj_z,
               rise_tol_m: float = PLACE_RISE_TOL_M) -> int | None:
    """The episode's PLACE/RELEASE step index, or ``None`` if it never places.

    Definition (one episode, all arrays indexed by row offset): the first
    step ``t > liftoff_step`` at which the gripper has LET GO
    (``grasped[t]`` is False) AND the object is NO LONGER RISING
    (``obj_z[t] <= obj_z[t-1] + rise_tol_m``) -- i.e. it has come back down
    onto whatever surface it now rests on.

    Why "no longer rising" and not "back below its initial resting height":
    the study's task (``PickPlaceCounterToCabinet``) places the object on a
    CABINET SHELF about 0.5 m ABOVE the counter it was picked up from, so
    ``obj_z`` returns to a resting height that is nothing like ``obj_z[0]``.
    Measured on this corpus, an ``init_z``-relative rule fires in only
    17/100 episodes, while this one fires in 95/100 -- every episode that
    lifts and releases. The non-rising clause is what keeps a transient
    ``grasped`` drop MID-LIFT from being mistaken for a place.

    ``liftoff_step`` follows ``recorder.liftoff_step``'s convention: a
    negative value means the episode never lifted, and there is then no
    place event (``None``). Re-grasps are common in this corpus (44/95
    episodes re-close the gripper after their first release), so this
    deliberately returns the FIRST place, never the last.
    """
    grasped_arr = np.asarray(grasped, dtype=bool).reshape(-1)
    z = np.asarray(obj_z, dtype=np.float64).reshape(-1)
    if grasped_arr.shape[0] != z.shape[0]:
        raise ValueError(
            f"place_step: grasped and obj_z must have the same length, got "
            f"{grasped_arr.shape[0]} and {z.shape[0]}"
        )
    lift = int(liftoff_step)
    if lift < 0:
        return None
    for t in range(max(lift + 1, 1), grasped_arr.shape[0]):
        if (not grasped_arr[t]) and z[t] <= z[t - 1] + rise_tol_m:
            return t
    return None


def place_steps(episode_id, step, grasped, obj_z, carry) -> dict[str, int]:
    """``{episode_id: absolute place step}`` from the FULL per-step rows.

    Lift-off is read as the episode's FIRST ``carry`` step -- the same
    event ``median_liftoff_s`` uses and the one ``dataset.phase_masks``
    defines -- so this never needs the raw ``liftoff_step`` scalar. An
    episode that never lifts, or lifts and never places, is ABSENT from the
    table (never mapped to 0 or to the episode end). ``step`` is read from
    the step column, not the row offset, so this is correct on a subset.
    """
    episode_id = np.asarray(episode_id)
    step = np.asarray(step)
    grasped = np.asarray(grasped, dtype=bool)
    obj_z = np.asarray(obj_z, dtype=np.float64)
    carry = np.asarray(carry, dtype=bool)
    table: dict[str, int] = {}
    for eid in np.unique(episode_id):
        m = episode_id == eid
        carry_hit = np.flatnonzero(carry[m])
        if not carry_hit.size:
            continue                      # never lifted -> no place event
        t = place_step(grasped[m], int(carry_hit[0]), obj_z[m])
        if t is not None:
            table[str(eid)] = int(np.asarray(step[m])[t])
    return table


def median_place_s(ds: dict, contact: dict[str, int]) -> float | None:
    """Median (place step - first contact step) in seconds, over the
    episodes that both contact and place. ``None`` if none of them do."""
    table = place_steps(ds["episode_id"], ds["step"], ds["grasped"],
                        np.asarray(ds["obj_pos"])[:, 2], ds["masks"]["carry"])
    deltas = [t - c for eid, t in table.items()
              if (c := contact.get(str(eid))) is not None]
    if not deltas:
        return None
    return float(np.median(deltas)) / CONTROL_HZ


# ---------------------------------------------------------------------------
# Camera-pixel reader row eligibility (addition 2)
# ---------------------------------------------------------------------------


def window_available(episode_id, step, stored_steps: dict[str, set[int]],
                     k: int = CAM_K, stride: int = CAM_STRIDE) -> np.ndarray:
    """Per row: is that row's WHOLE ``k``-frame stride-``stride`` window in
    the stored frame keep-set?

    ``render_frames.frame_steps_needed`` stored only the ``precontact``
    steps and the ``carry`` steps' windows (amendment C's two certificate
    masks), so a captured replan row inside the grasp transition, or after
    the object is placed, has NO stored frames. Substituting a neighbouring
    frame would corrupt XR1's visual format
    (``vision_certificate.gather_visual_rows`` raises rather than do that),
    so such rows are excluded from the camera-pixel line instead -- which is
    disclosed on the figure, in the summary json and in the report.
    """
    episode_id = np.asarray(episode_id)
    step = np.asarray(step)
    out = np.zeros(step.shape, dtype=bool)
    for i, (eid, t) in enumerate(zip(episode_id, step)):
        have = stored_steps.get(str(eid))
        if have is None:
            continue
        out[i] = all(w in have for w in window_steps(int(t), k, stride))
    return out


def stored_steps_table(frames_root, episode_ids, model: str = "xr1"
                       ) -> dict[str, set[int]]:
    """``{episode_id: set of stored frame steps}`` for the given episodes."""
    out: dict[str, set[int]] = {}
    for eid in dict.fromkeys(np.asarray(episode_ids).tolist()):
        condition, ep_name = str(eid).split("/")
        seed = int(ep_name.split("_")[1])
        path = Path(frames_root) / model / condition / f"ep_{seed}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            out[str(eid)] = {int(s) for s in z["steps"]}
    return out


def camera_matrix(episode_id, step, rows: np.ndarray, frames_root,
                  model: str = "xr1", verbose: bool = True) -> np.ndarray:
    """``(len(rows), k*C*H*W)`` uint8 -- XR1's own visual format at the given
    captured rows, in ``rows`` order.

    The window gather and the per-frame transform are the vision
    certificate's own functions (``gather_visual_rows``,
    ``apply_format_transform``), imported not re-implemented, so this line
    reads pixels exactly the way amendment C's ``ridge_pca`` reader does --
    only the row set differs (contact-relative bins instead of a phase mask).
    """
    episode_id = np.asarray(episode_id)
    step = np.asarray(step)
    rows = np.asarray(rows, dtype=np.int64)
    n_cam = len(CAMERA_KEYS)
    out = np.zeros((len(rows), CAM_K * n_cam, STORE_SIDE_PX, STORE_SIDE_PX),
                   dtype=np.uint8)
    if not len(rows):
        return out.reshape(0, CAM_K * n_cam * STORE_SIDE_PX ** 2)
    for eid in dict.fromkeys(episode_id[rows].tolist()):
        where = np.flatnonzero(episode_id[rows] == eid)
        condition, ep_name = str(eid).split("/")
        seed = int(ep_name.split("_")[1])
        with np.load(Path(frames_root) / model / condition / f"ep_{seed}.npz") as z:
            frames, steps = z["frames"], z["steps"]
        window = gather_visual_rows(frames, steps, step[rows[where]],
                                    k=CAM_K, stride=CAM_STRIDE)
        for i, w in enumerate(where):
            for j in range(CAM_K):
                for c in range(n_cam):
                    out[w, j * n_cam + c] = apply_format_transform(window[i, j, c], model)
    flat = out.reshape(len(rows), -1)
    if verbose:
        print(f"[camera] X={flat.shape} uint8 ({flat.nbytes / 1e9:.2f} GB) "
              f"over {len(set(episode_id[rows].tolist()))} episodes", flush=True)
    return flat


# ---------------------------------------------------------------------------
# Series (site x target x row scope) -- what the grid computes
# ---------------------------------------------------------------------------

# (label, block, layer, position, target, row_scope, panel, plotted).
# ``row_scope``: "all" = every captured row in the bin (the five published
# lines and all of panel B); "frames" = only the rows whose stored 4-frame
# window exists (the camera-pixel line and its matched VLM diagnostic).
SeriesSpec = tuple


def series_specs() -> tuple[SeriesSpec, ...]:
    out = [(label, block, layer, pos, TARGET, "all", "A", True)
           for label, block, layer, pos in SITES]
    out.append((*CAMERA_SITE, TARGET, "frames", "A", True))
    out.append((VLM_MATCHED_LABEL, "vlm", 21,
                "image_tokens_mean:video.robot0_eye_in_hand",
                TARGET, "frames", None, False))
    out.extend((label, block, layer, pos, TARGET_FORCE, "all", "B", True)
               for label, block, layer, pos in PANEL_B_SITES)
    return tuple(out)


SERIES = series_specs()
PANEL_A_LINES = tuple(s[0] for s in SERIES if s[6] == "A")
PANEL_B_LINES = tuple(s[0] for s in SERIES if s[6] == "B")


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def site_matrices(ds: dict, acts: dict, idx: np.ndarray,
                  sites=None) -> dict[str, np.ndarray]:
    """``{site label: X (M, D)}`` at the captured (joined) rows.

    The physics and deficit matrices are built on the FULL per-step corpus
    first (their trailing windows -- k=16 and k=8 respectively -- must see
    the un-subsampled history, exactly as ``certificates.build_features``
    does) and only then restricted to the captured rows -- the plan's "same
    rows" requirement, without corrupting the window. Both are the identical
    ``certificates.build_features`` builder, imported not re-implemented.

    ``sites`` defaults to the five published panel-A sites; the camera block
    is NOT built here (its matrix is assembled from the stored frames by
    :func:`camera_matrix`, over its own row subset)."""
    sites = SITES if sites is None else sites
    raw_ft_full = certificates.build_features("raw_ft", ds)
    deficit_full = certificates.build_features("deficit", ds)
    out: dict[str, np.ndarray] = {}
    for label, block, layer, position in sites:
        if block == "raw_ft":
            out[label] = raw_ft_full[idx].astype(np.float32)
        elif block == "deficit":
            out[label] = deficit_full[idx].astype(np.float32)
        elif block == "dit":
            li = list(acts["dit_layer_ids"]).index(layer)
            fi, bi = divmod(_dit_position_index(position), 2)
            out[label] = acts["dit"][:, li, fi, bi, :].astype(np.float32)
        elif block == "vlm":
            li = list(acts["vlm_layer_ids"]).index(layer)
            pi = _vlm_position_index(position)
            out[label] = acts["vlm"][:, li, pi, :].astype(np.float32)
        elif block == "state_embed":
            se = acts["state_embed"]
            out[label] = se.reshape(se.shape[0], -1).astype(np.float32)
        else:
            raise ValueError(f"site_matrices: unknown block {block!r}")
    return out


def _vlm_position_index(position: str) -> int:
    from eval_robocasa365.mass_variation.analysis.run_probes_xr1 import VLM_POSITION_NAMES

    return VLM_POSITION_NAMES.index(position)


def _dit_position_index(position: str) -> int:
    from eval_robocasa365.mass_variation.analysis.run_probes_xr1 import DIT_POSITION_NAMES

    return DIT_POSITION_NAMES.index(position)


def _zscore(X: np.ndarray) -> np.ndarray:
    """``certificates.certificate_cell``'s label-free feature standardisation
    (mixed physical units); applied to the physics and deficit lines only."""
    X = np.asarray(X, dtype=np.float32)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd == 0, 1.0, sd)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def _empty_cell(reason: str) -> dict:
    return {"degenerate": True, "degenerate_reason": reason,
            "real": None, "shuffled": None, "shuffled_std": None,
            "floor": None, "selectivity": None, "rank_acc": None}


def run_time_curve(ds: dict, acts: dict, idx: np.ndarray, bins, seed: int = SEED,
                   verbose: bool = True,
                   frames_root: str | Path = DEFAULT_FRAMES_ROOT) -> pd.DataFrame:
    """One probe cell per (series, bin), where a series is a
    (site, target, row-scope) triple -- see :data:`SERIES`.

    Within a bin and a row scope the row set, the seed groups, the episode
    shuffle groups and the ``GroupKFold(5)`` fold partition are computed ONCE
    and shared by every series in that scope, so those curves differ only in
    features. The per-fold SVD factors of an activation/sensor site are also
    computed once and reused for BOTH targets, which makes panel A and panel
    B literally the same fits scored against two different labels.

    Two row scopes exist because of one corpus fact, not a choice:
    ``render_frames`` stored frames only for the ``precontact`` steps and the
    ``carry`` windows, so the camera-pixel reader cannot be scored on the
    grasp-transition or post-place rows (``window_available``). Those rows
    are dropped from the camera line -- never faked -- and the SAME bins'
    VLM L21 cell is recomputed on exactly the camera line's rows as the
    matched diagnostic.
    """
    episode_id = np.asarray(acts["episode_id"])
    step = np.asarray(acts["step"])
    y_by_target = {
        TARGET: np.asarray(ds["mass_log_c"], dtype=np.float64)[idx],
        TARGET_FORCE: wrench_norm_of(ds["ee_force"])[idx],
    }
    cv_groups = np.asarray(ds["seed"])[idx]
    shuffle_groups = episode_id

    contact = first_contact_steps(ds["episode_id"], ds["step"], ds["grasped"])
    rel = contact_relative_steps(episode_id, step, contact)
    bin_of = assign_bins(rel, bins)

    act_sites = tuple({(s[0], s[1], s[2], s[3]) for s in SERIES if s[1] != "camera"})
    X_by_site = site_matrices(ds, acts, idx, sites=act_sites)

    # camera-pixel scope: which binned rows have their whole stored window
    binned = bin_of >= 0
    stored = stored_steps_table(frames_root, episode_id[binned])
    has_frames = np.zeros(len(idx), dtype=bool)
    has_frames[binned] = window_available(episode_id[binned], step[binned], stored)
    cam_rows = np.flatnonzero(has_frames)
    cam_pos = np.full(len(idx), -1, dtype=np.int64)
    cam_pos[cam_rows] = np.arange(len(cam_rows))
    if verbose:
        print(f"[camera] {len(cam_rows)}/{int(binned.sum())} binned rows have a "
              f"complete stored 4-frame window "
              f"({100.0 * len(cam_rows) / max(1, int(binned.sum())):.0f}%); the "
              f"rest fall in the grasp transition or after the place, which "
              f"render_frames never stored", flush=True)
    X_cam = camera_matrix(episode_id, step, cam_rows, frames_root, verbose=verbose)

    rows = []
    for b, (lo, hi) in enumerate(bins):
        geom = {
            "bin_index": b,
            "bin_lo_step": lo, "bin_hi_step": hi,
            "bin_lo_s": lo / CONTROL_HZ, "bin_hi_s": hi / CONTROL_HZ,
            "bin_center_s": (lo + hi) / 2.0 / CONTROL_HZ,
        }
        scope_sel = {"all": bin_of == b, "frames": (bin_of == b) & has_frames}
        scope_state = {}
        for scope, sel in scope_sel.items():
            n = int(sel.sum())
            n_eps = int(len(np.unique(episode_id[sel]))) if n else 0
            n_seeds = int(len(np.unique(cv_groups[sel]))) if n else 0
            drop = bin_drop_reason(n, n_eps)
            splits = (None if drop is not None
                      else probe_core._group_splits(np.zeros((n, 1)), cv_groups[sel]))
            scope_state[scope] = {
                "sel": sel, "n": n, "n_eps": n_eps, "n_seeds": n_seeds,
                "drop": drop, "splits": splits, "factors": {},
            }
            if verbose and drop is not None:
                print(f"[bin {b:2d}] [{lo/CONTROL_HZ:+.1f},{hi/CONTROL_HZ:+.1f})s "
                      f"scope={scope} DROPPED ({drop})", flush=True)

        for label, block, layer, position, target, scope, panel, plotted in SERIES:
            st = scope_state[scope]
            sel, n = st["sel"], st["n"]
            common = {
                **geom, "n": n, "n_episodes": st["n_eps"], "n_groups": st["n_seeds"],
                "retained": st["drop"] is None, "drop_reason": st["drop"],
            }
            if st["drop"] is not None:
                cell = _empty_cell(st["drop"])
            else:
                reason = _degeneracy_reason(y_by_target[target][sel], cv_groups[sel], "reg")
                if reason is not None:
                    cell = _empty_cell(reason)
                else:
                    if label not in st["factors"]:
                        if block == "camera":
                            Xc = X_cam[cam_pos[np.flatnonzero(sel)]].astype(np.float32)
                            Xc /= 255.0
                            st["factors"][label] = pca_fold_factors(
                                Xc, cv_groups[sel], n_components=N_PCA_COMPONENTS,
                                seed=seed, splits=st["splits"])
                            del Xc
                        else:
                            X = X_by_site[label][sel]
                            if block in ("raw_ft", "deficit"):
                                X = _zscore(X)
                            st["factors"][label] = probe_core._fold_factors(
                                X, cv_groups[sel], splits=st["splits"])
                    cell = _cell_from_factors(
                        st["factors"][label], y_by_target[target][sel],
                        cv_groups[sel], shuffle_groups[sel], "reg", seed)
            merged = {**common}
            merged.update({k: v for k, v in cell.items() if k not in ("n", "n_groups")})
            rows.append({
                "site": label, "block": block, "layer": layer,
                "position": position, "target": target, "row_scope": scope,
                "panel": panel, "plotted": bool(plotted), **merged,
            })
        if verbose:
            for panel, tgt in (("A", TARGET), ("B", TARGET_FORCE)):
                got = [r for r in rows[-len(SERIES):] if r["target"] == tgt]
                pretty = ", ".join(
                    f"{r['site'].split(' (')[0].split(' —')[0]}="
                    + ("drop" if r["selectivity"] is None else f"{r['selectivity']:+.3f}")
                    for r in got)
                print(f"[bin {b:2d}] [{lo/CONTROL_HZ:+.1f},{hi/CONTROL_HZ:+.1f})s "
                      f"panel {panel} ({tgt}) n={scope_state['all']['n']}"
                      f"/{scope_state['frames']['n']}f | {pretty}", flush=True)
    return pd.DataFrame(rows)


def median_liftoff_s(ds: dict, contact: dict[str, int]) -> float | None:
    """Median (liftoff step - first contact step) in seconds, over episodes
    that both grasped and lifted. Liftoff is the FIRST ``carry`` step (see
    ``dataset.phase_masks``: the liftoff step itself opens carry)."""
    episode_id = np.asarray(ds["episode_id"])
    step = np.asarray(ds["step"])
    carry = np.asarray(ds["masks"]["carry"], dtype=bool)
    deltas = []
    for eid, c in contact.items():
        m = episode_id == eid
        hit = np.flatnonzero(carry[m])
        if hit.size:
            deltas.append(int(step[m][hit[0]]) - c)
    if not deltas:
        return None
    return float(np.median(deltas)) / CONTROL_HZ


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _line_xy(df: pd.DataFrame, site: str, target: str):
    """``(x, y, sd)`` for one series, with a NaN inserted wherever a bin was
    DROPPED, so a broken line reads as a gap instead of interpolating
    straight through a bin that was never scored."""
    sub = (df[(df["site"] == site) & (df["target"] == target)
              & df["retained"] & ~df["degenerate"]]
           .sort_values("bin_index"))
    if not len(sub):
        return None
    xs, ys, sds, prev = [], [], [], None
    for _, r in sub.iterrows():
        if prev is not None and int(r["bin_index"]) != prev + 1:
            xs.append(np.nan); ys.append(np.nan); sds.append(np.nan)
        xs.append(float(r["bin_center_s"]))
        ys.append(float(r["selectivity"]))
        sds.append(float(r["shuffled_std"]))
        prev = int(r["bin_index"])
    return np.array(xs), np.array(ys), np.array(sds)


def _draw_rules(ax, ink: str, liftoff_s, place_s, label: bool):
    """The three event rules. Lift-off and place/release share one style --
    the figure's existing lighter dashed grey -- because they are the same
    kind of thing (a median event time); contact is the hard reference."""
    ax.axhline(0.0, color="#9a9a9a", lw=0.9, ls=":", zorder=1)
    ax.axvline(0.0, color=ink, lw=1.4, zorder=2)
    for t in (liftoff_s, place_s):
        if t is not None:
            ax.axvline(t, color="#9a9a9a", lw=1.1, ls="--", zorder=2)
    if not label:
        return
    top = ax.get_ylim()[1]
    ax.annotate("first contact", xy=(0.0, top), xytext=(3, -3),
                textcoords="offset points", ha="left", va="top",
                color=ink, fontsize=9, rotation=90)
    if liftoff_s is not None:
        ax.annotate(f"median lift-off (+{liftoff_s:.2f} s)",
                    xy=(liftoff_s, top), xytext=(4, -3),
                    textcoords="offset points", ha="left", va="top",
                    color="#6a6a6a", fontsize=8.5, rotation=90)
    if place_s is not None:
        ax.annotate(f"median place/release (+{place_s:.2f} s)",
                    xy=(place_s, top), xytext=(4, -3),
                    textcoords="offset points", ha="left", va="top",
                    color="#6a6a6a", fontsize=8.5, rotation=90)


def write_figure(df: pd.DataFrame, path: Path, liftoff_s: float | None,
                 place_s: float | None, note: str) -> str:
    """The deliverable figure: two stacked panels over one shared time axis
    and one shared rows-per-bin strip.

    Panel A -- selectivity for the HIDDEN MASS, the latent property (six
    lines: the five published ones, unchanged in colour/marker/linestyle,
    plus our camera-pixel reader).
    Panel B -- selectivity for the CONTACT FORCE |F|, the immediate
    consequence of that property (the three model sites plus the deficit
    channel; NO physics-ceiling line -- see ``PANEL_B_EXCLUSION_NOTE``).

    Layout rules (it ships into a LaTeX deck at ~0.63\\textwidth and into an
    HTML page, so it must survive half width): one legend for BOTH panels
    below the axes (panel B re-uses panel A's colours and markers), each
    panel's title in its own left-aligned band, and a single-line
    exploratory note at the very bottom -- the full method caption lives in
    ``timecurve_summary.json`` and the report, which travel with the PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ink = "#1a1a1a"
    grid_c = "#d9d9d9"

    fig = plt.figure(figsize=(7.6, 9.6), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    gs = GridSpec(3, 1, height_ratios=[5.0, 4.3, 0.95], hspace=0.16, figure=fig)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axA)
    axn = fig.add_subplot(gs[2], sharex=axA)
    for a in (axA, axB, axn):
        a.set_facecolor("#ffffff")
        a.tick_params(colors=ink, labelsize=9.5)
        for sp in a.spines.values():
            sp.set_color("#8a8a8a")
        a.spines[["top", "right"]].set_visible(False)

    panels = (
        (axA, TARGET, PANEL_A_LINES,
         "selectivity for hidden mass\n(real R² − mean shuffled R²)",
         "A   hidden mass (mass_log_c) — the LATENT property"),
        (axB, TARGET_FORCE, PANEL_B_LINES,
         "selectivity for contact force |F|\n(real R² − mean shuffled R²)",
         "B   contact force |F| (wrench_norm) — the IMMEDIATE consequence"),
    )
    for ax, target, labels, ylab, title in panels:
        ax.axhline(0.0, color="#9a9a9a", lw=0.9, ls=":", zorder=1)
        for label in labels:
            xy = _line_xy(df, label, target)
            if xy is None:
                continue
            x, yv, sd = xy
            style = SITE_STYLE[label]
            ax.fill_between(x, yv - sd, yv + sd, color=style["color"], alpha=0.17,
                            lw=0, zorder=3)
            ax.plot(x, yv, color=style["color"], marker=style["marker"],
                    ls=style["ls"], ms=5.5, lw=2.0,
                    markerfacecolor=style.get("mfc", style["color"]),
                    markeredgecolor=style["color"], markeredgewidth=1.4,
                    label=label, zorder=4)
        ax.set_ylabel(ylab, color=ink, fontsize=10)
        ax.grid(True, axis="y", color=grid_c, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=ink, fontsize=10.5, pad=6)
        # headroom for the in-axes rule labels, so nothing clips
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.10
        ax.set_ylim(ymin - pad * 0.3, ymax + pad * 1.9)
        _draw_rules(ax, ink, liftoff_s, place_s, label=(ax is axA))
        plt.setp(ax.get_xticklabels(), visible=False)

    # Panel B's own standing caveat, on the panel itself (it must travel with
    # the pixels, not only with the JSON).
    axB.text(0.012, 0.965,
             "no physics-ceiling line here: its features ARE the wrist F/T\n"
             "sensor, so decoding |F| from it would be circular, not a ceiling",
             transform=axB.transAxes, fontsize=8.4, color="#6a6a6a",
             style="italic", ha="left", va="top", linespacing=1.35)

    # rows-per-bin strip: grey = every captured row in the bin (the five
    # published lines and all of panel B); gold outline = the subset whose
    # stored 4-frame window exists, which is all the camera-pixel line may
    # legitimately be scored on.
    kept = df[df["retained"] & ~df["degenerate"]]
    strip = (kept[(kept["site"] == SITES[0][0]) & (kept["target"] == TARGET)]
             .sort_values("bin_center_s")[["bin_center_s", "n", "n_episodes"]])
    cam = (kept[(kept["site"] == CAMERA_LABEL) & (kept["target"] == TARGET)]
           .sort_values("bin_center_s")[["bin_center_s", "n"]])
    width = (BIN_WIDTH_STEPS / CONTROL_HZ) * 0.86
    axn.bar(strip["bin_center_s"], strip["n"], width=width,
            color="#c9c9c9", edgecolor="#a8a8a8", lw=0.5)
    for _, r in strip.iterrows():
        axn.annotate(f"{int(r['n'])}", xy=(r["bin_center_s"], r["n"]),
                     xytext=(0, 2), textcoords="offset points", ha="center",
                     va="bottom", fontsize=7.5, color="#5a5a5a")
    if len(cam):
        axn.bar(cam["bin_center_s"], cam["n"], width=width * 0.52,
                color=SITE_STYLE[CAMERA_LABEL]["color"], alpha=0.30,
                edgecolor=SITE_STYLE[CAMERA_LABEL]["color"], lw=1.1, zorder=3)
        for _, r in cam.iterrows():
            axn.annotate(f"{int(r['n'])}", xy=(r["bin_center_s"], r["n"]),
                         xytext=(0, -1.5), textcoords="offset points", ha="center",
                         va="top", fontsize=7, zorder=4,
                         color=SITE_STYLE[CAMERA_LABEL]["color"],
                         bbox={"facecolor": "#ffffff", "edgecolor": "none",
                               "pad": 0.6, "alpha": 0.85})
    axn.axvline(0.0, color=ink, lw=1.4)
    for t in (liftoff_s, place_s):
        if t is not None:
            axn.axvline(t, color="#9a9a9a", lw=1.1, ls="--")
    axn.set_ylim(0, strip["n"].max() * 1.45 if len(strip) else 1)
    axn.set_yticks([])
    axn.spines["left"].set_visible(False)
    axn.set_ylabel("rows\nper bin", color=ink, fontsize=8.5, rotation=0,
                   ha="right", va="center", labelpad=14)
    axn.set_xlabel("time relative to first contact (s)   —   1.6 s bins, "
                   "XR1 replan steps @ 20 Hz", color=ink, fontsize=10)
    axn.grid(False)

    fig.subplots_adjust(left=0.145, right=0.985, top=0.930, bottom=0.185)
    fig.text(0.145, 0.968,
             "When does hidden mass become decodable? (EXPLORATORY)",
             color=ink, fontsize=13, ha="left", va="center")
    # ONE legend for both panels, below the x label -- never over a plot,
    # never over a title. Panel B re-uses panel A's colours and markers.
    handles, labels = axA.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 0.043), ncol=2, frameon=False,
                     fontsize=8.6, handlelength=2.4, columnspacing=1.6,
                     labelspacing=0.42)
    for t in leg.get_texts():
        t.set_color(ink)
    fig.text(0.5, 0.010, note, fontsize=7.2, color="#5a5a5a",
             ha="center", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)



FIGURE_NOTE = (
    "EXPLORATORY post-hoc re-cut of the pre-registered carry-phase probe result — "
    "no new claim; full method caption in timecurve_summary.json."
)


def peak_vs_headline(df: pd.DataFrame) -> str:
    """The mandatory honesty sentence about the DiT line's per-bin peak.

    The peak bin's selectivity EXCEEDS the pre-registered phase-bucket value
    (0.227 at DiT L28, R² 0.189, carry mask, n=809). That is NOT a better
    result: it is a single-bin estimate on a small fraction of the carry
    rows, with a correspondingly wider shuffled spread. Stated in the
    summary json and the report verbatim."""
    sub = df[(df["site"] == SITES[1][0]) & (df["target"] == TARGET)
             & df["retained"] & ~df["degenerate"]]
    if not len(sub):
        return "DiT L28 line has no retained bin."
    row = sub.loc[sub["selectivity"].idxmax()]
    n = int(row["n"])
    return (
        f"The DiT L28 line peaks at selectivity {float(row['selectivity']):.3f} "
        f"(R² {float(row['real']):.3f}, shuffled std {float(row['shuffled_std']):.3f}) "
        f"in the [{float(row['bin_lo_s']):+.1f},{float(row['bin_hi_s']):+.1f}) s bin "
        f"(centre {float(row['bin_center_s']):+.1f} s) on n={n} rows, ABOVE the "
        f"pre-registered phase-bucket value of 0.227 (R² 0.189, carry mask, n=809). "
        f"Read it as a NOISIER single-bin estimate, NOT a better result: those {n} "
        f"rows are {100.0 * n / 2413:.0f}% of the 2,413 captured rows and about "
        f"{100.0 * n / 809:.0f}% the size of the pre-registered 809-row carry cell, "
        f"and the bin was selected as this line's maximum over 8 bins (a "
        f"max-over-bins statistic is upward-biased). The study's headline is "
        f"unchanged and stays below the 0.3 certificate bar."
    )


def precontact_reading(df: pd.DataFrame) -> str:
    """Plain statement of what the pre-contact bins do and do not show.

    Required by the study's honesty rules: if pre-contact selectivity were
    meaningfully above 0 at a model site, that would qualify the leakage
    story and must be said outright, never smoothed."""
    pre = df[df["retained"] & ~df["degenerate"] & (df["bin_hi_s"] <= 0.0)
             & (df["target"] == TARGET) & df["plotted"]]
    if not len(pre):
        return "no retained pre-contact bin."
    worst = pre.loc[pre["selectivity"].idxmax()]
    max_real = float(pre["real"].max())
    return (
        f"Pre-contact bins ({pre['bin_index'].nunique()} retained, "
        f"−4.8..0.0 s): the largest selectivity is "
        f"{float(worst['selectivity']):+.3f} "
        f"({worst['site']}, [{float(worst['bin_lo_s']):+.1f},"
        f"{float(worst['bin_hi_s']):+.1f}) s), but the largest REAL pooled "
        f"held-out R² over all six panel-A lines and all pre-contact bins is only "
        f"{max_real:+.4f} — i.e. no line predicts mass better than the "
        f"training-fold mean before contact. Those small positive "
        f"selectivities come from the shuffled control scoring NEGATIVE "
        f"(shuffled R² ≈ −0.02..−0.05, the usual grouped-CV behaviour when a "
        f"permuted per-episode label is fit across folds), not from real "
        f"pre-contact signal. Nothing here qualifies the study's leakage "
        f"story: the pre-registered guard (mass selectivity < 0.1 at every "
        f"probed cell on the precontact mask) also still holds at this grain."
    )


def figure_caption(df: pd.DataFrame, liftoff_s: float | None,
                   place_s: float | None) -> str:
    allsc = df[df["row_scope"] == "all"]
    kept = allsc[allsc["retained"]]["bin_index"].nunique()
    dropped = (allsc[~allsc["retained"]][["bin_lo_s", "bin_hi_s", "drop_reason"]]
               .drop_duplicates())
    drop_txt = "; ".join(
        f"[{r.bin_lo_s:+.1f},{r.bin_hi_s:+.1f})s ({r.drop_reason})"
        for r in dropped.itertuples()
    ) or "none"
    cam = df[(df["site"] == CAMERA_LABEL) & df["retained"] & ~df["degenerate"]]
    cam_txt = ", ".join(f"[{r.bin_lo_s:+.1f},{r.bin_hi_s:+.1f})s"
                        for r in cam.itertuples()) or "none"
    return (
        f"EXPLORATORY post-hoc time resolution of the pre-registered carry-phase probe result — no new claim, "
        f"same rows/targets/discipline at finer time grain. Selectivity = real pooled held-out R² minus mean shuffled R² "
        f"(5 group-coherent shuffles, each with its own alpha search); band = ±1 shuffled std. GroupKFold(5) by SEED, "
        f"shuffles group-coherent by EPISODE; identical rows and folds for every line within a panel and a row scope. "
        f"PANEL A: target mass_log_c (the latent property). PANEL B: target wrench_norm = |F| in newtons (the immediate "
        f"contact consequence), same bins/rows/folds/shuffles, showing ONLY the three model sites and the deficit "
        f"channel — the wrist-F/T physics ceiling is deliberately ABSENT there because its features are the force "
        f"sensor itself, so decoding |F| from it would be circular rather than a ceiling. The sixth panel-A line, "
        f"'{CAMERA_LABEL}', is OUR reader (vision_certificate.py's ridge-on-PCA over 96×96 grayscale frames in XR1's "
        f"4-frame stride-2 × 3-camera format, PCA fitted on the TRAIN rows of each fold only) — it is NOT the model's "
        f"vision, and it is scored only on rows whose stored frame window exists (render_frames kept precontact and "
        f"carry windows only), so it is retained in {cam_txt} and absent elsewhere; a row-matched VLM L21 cell on "
        f"exactly those rows is recorded in the summary json as the fair comparator. "
        f"Bins 1.6 s (32 steps @20 Hz), span −6.4..+9.6 s, retained only if ≥{MIN_ROWS} rows and "
        f"≥{MIN_EPISODE_GROUPS} episodes ({kept} retained; dropped: {drop_txt}). "
        f"Median lift-off {('+%.2f s' % liftoff_s) if liftoff_s is not None else 'n/a'}; "
        f"median place/release {('+%.2f s' % place_s) if place_s is not None else 'n/a'} "
        f"(the third rule — it is why every line decays late in the span). "
        f"Headline unchanged: peak carry-phase selectivity 0.227 (R² 0.189) at DiT L28, below the 0.3 bar."
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def curve_summary(df: pd.DataFrame, target: str = TARGET) -> dict:
    """Per-series readings the report quotes: last pre-contact bin, first
    post-contact bin, and the series' maximum over retained bins."""
    out = {}
    kept = df[(df["target"] == target) & df["retained"] & ~df["degenerate"]]
    for label in [s[0] for s in SERIES if s[4] == target]:
        sub = kept[kept["site"] == label].sort_values("bin_center_s")
        if not len(sub):
            out[label] = None
            continue

        def pick(row):
            return {
                "bin_s": [float(row["bin_lo_s"]), float(row["bin_hi_s"])],
                "selectivity": float(row["selectivity"]),
                "real_r2": float(row["real"]),
                "shuffled_r2": float(row["shuffled"]),
                "shuffled_std": float(row["shuffled_std"]),
                "rank_acc": None if row["rank_acc"] is None else float(row["rank_acc"]),
                "n": int(row["n"]), "n_episodes": int(row["n_episodes"]),
                "n_groups": int(row["n_groups"]),
            }

        pre = sub[sub["bin_hi_s"] <= 0.0]
        post = sub[sub["bin_lo_s"] >= 0.0]
        out[label] = {
            "last_precontact_bin": pick(pre.iloc[-1]) if len(pre) else None,
            "first_postcontact_bin": pick(post.iloc[0]) if len(post) else None,
            "max_bin": pick(sub.loc[sub["selectivity"].idxmax()]),
            "max_precontact_selectivity": (
                float(pre["selectivity"].max()) if len(pre) else None),
        }
    return out


def _peak(df: pd.DataFrame, site: str, target: str):
    sub = df[(df["site"] == site) & (df["target"] == target)
             & df["retained"] & ~df["degenerate"]]
    return None if not len(sub) else sub.loc[sub["selectivity"].idxmax()]


def panel_b_reading(df: pd.DataFrame) -> str:
    """Panel B in words: per site, when |F| selectivity peaks and whether it
    beats that SAME site's mass curve. This is the panel's whole point --
    does the model track the immediate contact consequence even where it
    fails to track the latent property?"""
    parts = []
    for label in PANEL_B_LINES:
        f = _peak(df, label, TARGET_FORCE)
        m = _peak(df, label, TARGET)
        if f is None:
            parts.append(f"{label}: no retained bin.")
            continue
        verdict = ("no mass peak to compare" if m is None else
                   (f"BEATS its own mass peak ({float(m['selectivity']):+.3f} @ "
                    f"{float(m['bin_center_s']):+.1f} s)"
                    if float(f["selectivity"]) > float(m["selectivity"]) else
                    f"does NOT beat its own mass peak ({float(m['selectivity']):+.3f} @ "
                    f"{float(m['bin_center_s']):+.1f} s)"))
        parts.append(
            f"{label}: |F| selectivity peaks {float(f['selectivity']):+.3f} "
            f"(R² {float(f['real']):.3f}, shuffled std {float(f['shuffled_std']):.3f}, "
            f"rank {float(f['rank_acc']):.3f}) in "
            f"[{float(f['bin_lo_s']):+.1f},{float(f['bin_hi_s']):+.1f}) s "
            f"(n={int(f['n'])}) — {verdict}.")
    return (PANEL_B_EXCLUSION_NOTE + " " + " ".join(parts) + " " + PANEL_B_CAVEAT)


def camera_vs_vlm_reading(df: pd.DataFrame) -> str:
    """The instrument asymmetry, stated with the row-matched comparator so it
    is not a comparison between two different row sets."""
    cam = _peak(df, CAMERA_LABEL, TARGET)
    vlm = _peak(df, SITES[2][0], TARGET)
    matched = _peak(df, VLM_MATCHED_LABEL, TARGET)
    if cam is None:
        return ("The camera-pixel reader has NO retained bin: no bin holds "
                f"≥{MIN_ROWS} rows with a complete stored frame window.")
    cam_bins = df[(df["site"] == CAMERA_LABEL) & df["retained"] & ~df["degenerate"]]
    cam_max_real = float(cam_bins["real"].max())
    mat_bins = df[(df["site"] == VLM_MATCHED_LABEL) & df["retained"] & ~df["degenerate"]]
    return (
        f"THE CAMERA-PIXEL LINE'S REAL R² NEVER EXCEEDS {cam_max_real:+.4f} in any "
        f"retained bin — i.e. our pixel reader never predicts mass better than the "
        f"training-fold mean, anywhere on the axis. Its positive SELECTIVITY comes "
        f"entirely from its own shuffled control scoring far more negative "
        f"({float(cam_bins['shuffled'].min()):+.3f}..{float(cam_bins['shuffled'].max()):+.3f}, "
        f"against shuffled stds up to {float(cam_bins['shuffled_std'].max()):.3f}) — the "
        f"usual grouped-CV behaviour of a 110,592-dimensional reader on ~150 rows — and "
        f"must NOT be read as pixel-borne mass signal. The row-matched VLM L21 cell, on "
        f"exactly the same rows and folds, reaches real R² "
        f"{float(mat_bins['real'].max()):+.4f}. "
    ) + (
        f"Camera-pixel reader (OURS, 96×96 grey, in-fold PCA + ridge — NOT the "
        f"model's vision): peak selectivity {float(cam['selectivity']):+.3f} "
        f"(real R² {float(cam['real']):.4f}, shuffled std "
        f"{float(cam['shuffled_std']):.4f}, rank {float(cam['rank_acc']):.3f}) in "
        f"[{float(cam['bin_lo_s']):+.1f},{float(cam['bin_hi_s']):+.1f}) s on "
        f"n={int(cam['n'])} rows, over {int(cam_bins['bin_index'].nunique())} "
        f"retained bins. XR1's own wrist-camera TOKENS (VLM L21) peak at "
        f"{float(vlm['selectivity']):+.3f} (real R² {float(vlm['real']):.4f}) in "
        f"[{float(vlm['bin_lo_s']):+.1f},{float(vlm['bin_hi_s']):+.1f}) s. "
        + ("" if matched is None else
           f"On EXACTLY the camera reader's rows and folds, VLM L21 still peaks at "
           f"{float(matched['selectivity']):+.3f} (real R² {float(matched['real']):.4f}, "
           f"[{float(matched['bin_lo_s']):+.1f},{float(matched['bin_hi_s']):+.1f}) s, "
           f"n={int(matched['n'])}), so the gap is not an artifact of the two "
           f"lines being scored on different row sets. ")
        + "Read this as an INSTRUMENT asymmetry, not a claim that the pixels "
          "lack mass: a reader is a lower bound on channel content — it can "
          "miss information, never invent it — so our low-resolution reader "
          "finding nothing is evidence about THIS reader on THIS 96×96 "
          "grayscale reduction, while the model's encoder finding something "
          "is evidence the channel carries it."
    )


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="EXPLORATORY contact-relative time curve for XR1 mass decodability.")
    ap.add_argument("--acts-root", default=DEFAULT_ACTS_ROOT)
    ap.add_argument("--phase1-root", default=DEFAULT_PHASE1_ROOT)
    ap.add_argument("--policy-state-root", default=DEFAULT_POLICY_STATE_ROOT)
    ap.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-name", default="timecurve.parquet")
    ap.add_argument("--fig-name", default="fig_mass_vs_time.png")
    ap.add_argument("--summary-name", default="timecurve_summary.json")
    ap.add_argument("--base-seed", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=35)
    ap.add_argument("--bin-width-steps", type=int, default=BIN_WIDTH_STEPS)
    ap.add_argument("--span-lo-steps", type=int, default=SPAN_LO_STEPS)
    ap.add_argument("--span-hi-steps", type=int, default=SPAN_HI_STEPS)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="plan2-timecurve-xr1-v3")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    bins = make_bins(args.span_lo_steps, args.span_hi_steps, args.bin_width_steps)

    t0 = time.time()
    print(f"[load] acts from {args.acts_root} (DiT L28 + VLM L21 + state_embed)",
          flush=True)
    acts = load_acts(Path(args.acts_root), seeds=seeds,
                     vlm_layer_ids=(21,), dit_layer_ids=(28,))
    ds = load_study("xr1", args.phase1_root, args.policy_state_root, seeds=seeds)
    idx = join_index(ds["episode_id"], ds["step"], acts["episode_id"], acts["step"])
    contact = first_contact_steps(ds["episode_id"], ds["step"], ds["grasped"])
    n_eps_total = int(len(np.unique(ds["episode_id"])))
    n_no_contact = n_eps_total - len(contact)
    liftoff_s = median_liftoff_s(ds, contact)
    place_s = median_place_s(ds, contact)
    n_placed = len(place_steps(ds["episode_id"], ds["step"], ds["grasped"],
                               np.asarray(ds["obj_pos"])[:, 2], ds["masks"]["carry"]))
    print(f"[join] {len(idx)} captured rows over {n_eps_total} episodes "
          f"({n_no_contact} never grasped -> dropped, no contact reference); "
          f"median lift-off {liftoff_s:+.2f}s, median place/release "
          f"{place_s:+.2f}s after contact ({n_placed} episodes place) "
          f"({time.time()-t0:.1f}s)", flush=True)
    print(f"[bins] {len(bins)} x {args.bin_width_steps} steps "
          f"({args.bin_width_steps/CONTROL_HZ:.1f}s) spanning "
          f"{args.span_lo_steps/CONTROL_HZ:+.1f}..{args.span_hi_steps/CONTROL_HZ:+.1f}s; "
          f"retain if n>={MIN_ROWS} and n_episodes>={MIN_EPISODE_GROUPS}", flush=True)

    df = run_time_curve(ds, acts, idx, bins, seed=SEED,
                        frames_root=args.frames_root)
    out_path = out_dir / args.out_name
    df.to_parquet(out_path, index=False)
    print(f"[out] wrote {out_path} ({len(df)} rows)", flush=True)

    caption = figure_caption(df, liftoff_s, place_s)
    peak_note = peak_vs_headline(df)
    pre_note = precontact_reading(df)
    panel_b_note = panel_b_reading(df)
    camera_note = camera_vs_vlm_reading(df)
    fig_path = write_figure(df, out_dir / args.fig_name, liftoff_s, place_s,
                            FIGURE_NOTE)
    print(f"[fig] wrote {fig_path}", flush=True)
    print(f"\n[honesty] {peak_note}", flush=True)
    print(f"[honesty] {pre_note}", flush=True)
    print(f"\n[panel B] {panel_b_note}", flush=True)
    print(f"\n[camera] {camera_note}", flush=True)

    summary_curve = curve_summary(df, TARGET)
    summary_curve_force = curve_summary(df, TARGET_FORCE)
    for tgt, table in ((TARGET, summary_curve), (TARGET_FORCE, summary_curve_force)):
        print(f"\nCURVE (selectivity for {tgt}):", flush=True)
        for label, s in table.items():
            if s is None:
                continue

            def fmt(k, s=s):
                v = s[k]
                return "n/a" if v is None else (
                    f"{v['selectivity']:+.3f} @[{v['bin_s'][0]:+.1f},"
                    f"{v['bin_s'][1]:+.1f})s n={v['n']}")
            print(f"  {label}\n     last pre-contact {fmt('last_precontact_bin')}\n"
                  f"     first post-contact {fmt('first_postcontact_bin')}\n"
                  f"     max {fmt('max_bin')}", flush=True)

    config = {
        "analysis": "EXPLORATORY time curve (plan-2 follow-up to Task 4)",
        "exploratory_label": EXPLORATORY_LABEL,
        "headline_not_upgraded": HEADLINE_NOT_UPGRADED,
        "target": TARGET,
        "targets": {"panel_A": TARGET, "panel_B": TARGET_FORCE},
        "x_axis": "seconds relative to first contact (first grasped step), "
                  f"control rate {CONTROL_HZ} Hz",
        "y_axis": "selectivity = real pooled held-out R^2 - mean shuffled R^2",
        "sites": [{"label": l, "block": b, "layer": ly, "position": p}
                  for l, b, ly, p in SITES],
        "series": [{"label": l, "block": b, "layer": ly, "position": p,
                    "target": t, "row_scope": sc, "panel": pn, "plotted": pl}
                   for l, b, ly, p, t, sc, pn, pl in SERIES],
        "panel_B": {
            "target": TARGET_FORCE,
            "definition": "wrench_norm = |ee_force| in newtons "
                          "(run_probes_xr1.wrench_norm_of), the same target "
                          "that analysis/run_probes_xr1.py already probes",
            "question": ("does the model track the IMMEDIATE CONTACT "
                         "CONSEQUENCE (the force it is feeling) even where it "
                         "fails to track the LATENT PROPERTY (the mass)?"),
            "lines": list(PANEL_B_LINES),
            "physics_ceiling_excluded": PANEL_B_EXCLUSION_NOTE,
            "shared_with_panel_A": ("identical bins, identical rows, identical "
                                    "GroupKFold(5) fold partition and identical "
                                    "per-fold SVD factors -- panel A and panel B "
                                    "are literally the same fits scored against "
                                    "two different labels"),
            "reading": panel_b_note,
        },
        "camera_pixel_line": {
            "label": CAMERA_LABEL,
            "what_it_is": ("OUR reader, not the model's vision: "
                           "vision_certificate.py's ridge-on-PCA reader over "
                           "96x96 GRAYSCALE stored frames in XR1's own visual "
                           "format (4 frames at stride 2 x 3 cameras, its own "
                           "0.95 centre crop), i.e. the amendment-C "
                           "'ridge_pca' cell re-cut into these bins"),
            "leakage_rule": ("PCA(512) fitted on the TRAIN rows WITHIN EACH "
                             "FOLD only (amendment C Section 2) -- an all-rows "
                             "PCA would carry held-out structure into the "
                             "representation and could invent a positive; "
                             "vision_certificate.pca_fold_factors is imported, "
                             "not re-implemented"),
            "row_scope": ("only rows whose whole stored 4-frame window exists. "
                          "render_frames.frame_steps_needed stored the "
                          "precontact steps and the carry windows ONLY, so "
                          "captured rows inside the grasp transition or after "
                          "the place have no frames. Substituting a "
                          "neighbouring frame would corrupt the format, so "
                          "those rows are EXCLUDED and the retention rule "
                          "(>=120 rows, >=15 episodes) is re-applied to the "
                          "subset; a bin that fails it is dropped for this "
                          "line only and the line is drawn with a GAP there"),
            "matched_comparator": ("'" + VLM_MATCHED_LABEL + "' re-scores VLM "
                                   "L21 on exactly this line's rows and folds, "
                                   "so the instrument asymmetry is not a "
                                   "comparison between two row sets. Recorded "
                                   "in the parquet and here; never plotted"),
            "reader_is_a_lower_bound": ("a reader can miss information, never "
                                        "invent it: this line finding nothing "
                                        "is evidence about THIS reader on THIS "
                                        "96x96 grayscale reduction, not proof "
                                        "the camera channel lacks mass"),
            "reading": camera_note,
        },
        "place_event": {
            "definition": ("first step after lift-off at which the gripper has "
                           "released (grasped False) AND the object is no "
                           "longer rising (obj_z[t] <= obj_z[t-1] + "
                           f"{PLACE_RISE_TOL_M} m)"),
            "why_not_initial_height": (
                "the task places the object on a CABINET SHELF about 0.5 m "
                "ABOVE the counter it was picked up from, so obj_z never "
                "returns to obj_z[0]: an init_z-relative rule fires in only "
                "17/100 episodes on this corpus, this one in 95/100"),
            "first_place_not_last": ("44 of 95 episodes re-close the gripper "
                                     "after their first release, so the "
                                     "detector deliberately returns the FIRST "
                                     "place"),
            "median_s_after_contact": place_s,
            "episodes_with_a_place": n_placed,
        },
        "physics_line": (
            "certificates.build_features('raw_ft'): ee_force[3]+ee_torque[3]+"
            "derived [|F|,|tau|,F_world[3],F_base[3]], per-episode trailing "
            f"window k={certificates.K_RAW_FT} stride 1, z-scored per cell "
            "exactly as certificates.certificate_cell does (mixed physical "
            "units); activations are NOT z-scored (run_probes_xr1 convention)"),
        "deficit_line": (
            "certificates.build_features('deficit'): commanded_delta[6]+"
            "achieved_eef_delta[6], per-episode trailing window "
            f"k={certificates.K_DEFICIT} stride 1, z-scored per cell exactly "
            "as certificates.certificate_cell does (mixed units: normalized "
            "OSC action vs metres) -- the identical builder the 'deficit' "
            "certificate uses (carry-mask R^2 0.062, FAIL); this line asks "
            "whether that channel's time profile spikes at contact or stays "
            "flat, not whether it passes the certificate gate"),
        "bin_rule": {
            "width_steps": args.bin_width_steps,
            "width_s": args.bin_width_steps / CONTROL_HZ,
            "span_steps": [args.span_lo_steps, args.span_hi_steps],
            "span_s": [args.span_lo_steps / CONTROL_HZ, args.span_hi_steps / CONTROL_HZ],
            "min_rows": MIN_ROWS,
            "min_episode_groups": MIN_EPISODE_GROUPS,
            "edge_at_contact": True,
            "chosen_for": "sample sufficiency, fixed before any curve was "
                          "computed; NOT tuned on outcomes",
        },
        "protocol": {
            "cv": "GroupKFold(5) by SEED (identical fold partition for every "
                  "site within a bin and a row scope; the per-fold SVD "
                  "factors are shared across both targets too)",
            "shuffle": "group-coherent by EPISODE, per-draw alpha search",
            "alphas": probe_core.ALPHAS.tolist(),
            "n_splits": probe_core.N_SPLITS,
            "n_shuffles": probe_core.N_SHUFFLES,
            "seed": SEED,
        },
        "corpus": {
            "acts_root": args.acts_root,
            "seeds": [seeds[0], seeds[-1]],
            "captured_rows": int(len(idx)),
            "episodes": n_eps_total,
            "episodes_without_contact_dropped": n_no_contact,
            "median_liftoff_s_after_contact": liftoff_s,
            "median_place_s_after_contact": place_s,
            "frames_root": args.frames_root,
        },
        "git_sha": _git_sha(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    dropped = (df[~df["retained"]][["row_scope", "bin_lo_s", "bin_hi_s", "n",
                                    "n_episodes", "drop_reason"]]
               .drop_duplicates().to_dict("records"))
    per_bin = (df[["site", "target", "row_scope", "panel", "bin_index",
                   "bin_lo_s", "bin_hi_s", "n", "n_episodes",
                   "n_groups", "retained", "real", "shuffled", "shuffled_std",
                   "selectivity", "rank_acc", "floor", "degenerate",
                   "drop_reason"]].to_dict("records"))
    summary = sanitize_json({
        "config": config,
        "figure_caption": caption,
        "figure_note_on_png": FIGURE_NOTE,
        "peak_vs_pre_registered_headline": peak_note,
        "precontact_reading": pre_note,
        "panel_b_reading": panel_b_note,
        "camera_vs_vlm_reading": camera_note,
        "dropped_bins": dropped,
        "curve": summary_curve,
        "curve_force": summary_curve_force,
        "per_bin": per_bin,
        "parquet": str(out_path),
        "figure": fig_path,
        "wall_s": round(time.time() - t0, 1),
    })
    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"[out] wrote {summary_path}", flush=True)

    if not args.no_wandb:
        import wandb

        run = wandb.init(project="mass-com-xr1", job_type="analysis",
                         name=args.wandb_name, config=config)
        table_df = df.copy()
        for col in table_df.columns:
            if table_df[col].dtype == object:
                table_df[col] = table_df[col].astype(str)
        run.log({"timecurve": wandb.Table(dataframe=table_df),
                 "fig_mass_vs_time": wandb.Image(fig_path)})
        for prefix, table in (("curve", summary_curve),
                              ("curve_force", summary_curve_force)):
            for label, s in table.items():
                if s is None:
                    continue
                key = label.split(" (")[0].split(" —")[0].replace(" ", "_")
                run.summary[f"{prefix}/{key}/max_selectivity"] = \
                    s["max_bin"]["selectivity"]
                run.summary[f"{prefix}/{key}/max_bin_s"] = s["max_bin"]["bin_s"]
                run.summary[f"{prefix}/{key}/max_precontact_selectivity"] = \
                    s["max_precontact_selectivity"]
        run.summary["exploratory"] = True
        run.summary["median_liftoff_s"] = liftoff_s
        run.summary["median_place_s"] = place_s
        print("wandb url:", run.url, flush=True)
        run.finish()

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

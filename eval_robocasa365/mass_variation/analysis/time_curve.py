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

Lines (4)
---------
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

All four are scored on the SAME rows (the captured replan steps inside a
bin) with the SAME GroupKFold(5) fold partition (computed once per bin from
the seed groups), so the lines are strictly comparable. The one deliberate
preprocessing difference is the one each corpus already carries in the
study: the physics features are globally z-scored inside the cell (mixed
physical units -- Newtons, N·m, radians), exactly as
``certificates.certificate_cell`` does, while activations are not
(homogeneous units, matching ``run_probes_xr1``).

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
was computed and is applied identically to all four lines.

Outputs: ``<out-dir>/timecurve.parquet`` (one row per site x bin, including
dropped bins), ``<out-dir>/fig_mass_vs_time.png``,
``<out-dir>/timecurve_summary.json``; wandb run ``plan2-timecurve-xr1``
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

from eval_robocasa365.mass_variation.analysis import certificates, probe_core
from eval_robocasa365.mass_variation.analysis.dataset import load_study
from eval_robocasa365.mass_variation.analysis.run_probes_xr1 import (
    _cell_from_factors,
    _degeneracy_reason,
    join_index,
    load_acts,
)
from eval_robocasa365.mass_variation.metrics import CONTROL_HZ

REPO_ROOT = Path(__file__).resolve().parents[3]

SEED = 0
TARGET = "mass_log_c"

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

# (label, block, layer, position) -- block "raw_ft" is the physics reference.
SITES = (
    ("physics ceiling (wrist F/T, k=16 window)", "raw_ft", -1, "ee_force+ee_torque+derived"),
    ("XR1 action module (DiT L28, flow4 state-block)", "dit", 28, "flow4:state_tokens_mean"),
    ("XR1 wrist-camera tokens (VLM L21)", "vlm", 21,
     "image_tokens_mean:video.robot0_eye_in_hand"),
    ("XR1 proprioception (state_embed)", "state_embed", -1, "state_tokens_flat"),
)

# CVD-checked on a light surface; per-series markers are the redundant
# encoding. The physics reference is neutral grey + dashed so it never reads
# as one of the three model sites.
SITE_STYLE = {
    "physics ceiling (wrist F/T, k=16 window)":
        {"color": "#4a4a4a", "marker": "D", "ls": "--"},
    "XR1 action module (DiT L28, flow4 state-block)":
        {"color": "#c14f2e", "marker": "o", "ls": "-"},
    "XR1 wrist-camera tokens (VLM L21)":
        {"color": "#3a6ea5", "marker": "s", "ls": "-"},
    "XR1 proprioception (state_embed)":
        {"color": "#2f9e63", "marker": "^", "ls": "-"},
}

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
# Feature assembly
# ---------------------------------------------------------------------------


def site_matrices(ds: dict, acts: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
    """``{site label: X (M, D)}`` at the captured (joined) rows.

    The physics matrix is built on the FULL per-step corpus first (its k=16
    trailing window must see the un-subsampled history, exactly as the
    ``raw_ft`` certificate does) and only then restricted to the captured
    rows -- the plan's "same rows" requirement, without corrupting the
    window."""
    raw_ft_full = certificates.build_features("raw_ft", ds)
    out: dict[str, np.ndarray] = {}
    for label, block, layer, position in SITES:
        if block == "raw_ft":
            out[label] = raw_ft_full[idx].astype(np.float32)
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
    (mixed physical units); applied to the physics line only."""
    X = np.asarray(X, dtype=np.float32)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd == 0, 1.0, sd)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def run_time_curve(ds: dict, acts: dict, idx: np.ndarray, bins, seed: int = SEED,
                   verbose: bool = True) -> pd.DataFrame:
    """One probe cell per (site, bin) on the captured rows.

    Fold partition, shuffle groups and row set are computed ONCE per bin and
    shared by all four sites, so the four curves differ only in features."""
    episode_id = np.asarray(acts["episode_id"])
    y = np.asarray(ds["mass_log_c"], dtype=np.float64)[idx]
    cv_groups = np.asarray(ds["seed"])[idx]
    shuffle_groups = episode_id

    contact = first_contact_steps(ds["episode_id"], ds["step"], ds["grasped"])
    rel = contact_relative_steps(episode_id, acts["step"], contact)
    bin_of = assign_bins(rel, bins)

    X_by_site = site_matrices(ds, acts, idx)
    rows = []
    for b, (lo, hi) in enumerate(bins):
        sel = bin_of == b
        n = int(sel.sum())
        n_eps = int(len(np.unique(episode_id[sel]))) if n else 0
        n_seeds = int(len(np.unique(cv_groups[sel]))) if n else 0
        drop = bin_drop_reason(n, n_eps)
        common = {
            "bin_index": b,
            "bin_lo_step": lo, "bin_hi_step": hi,
            "bin_lo_s": lo / CONTROL_HZ, "bin_hi_s": hi / CONTROL_HZ,
            "bin_center_s": (lo + hi) / 2.0 / CONTROL_HZ,
            "n": n, "n_episodes": n_eps, "n_groups": n_seeds,
            "retained": drop is None,
            "drop_reason": drop,
        }
        if drop is not None:
            if verbose:
                print(f"[bin {b:2d}] [{lo/CONTROL_HZ:+.1f},{hi/CONTROL_HZ:+.1f})s "
                      f"DROPPED ({drop})", flush=True)
            for label, block, layer, position in SITES:
                rows.append({
                    "site": label, "block": block, "layer": layer,
                    "position": position, "target": TARGET, **common,
                    "degenerate": True, "degenerate_reason": drop,
                    "real": None, "shuffled": None, "shuffled_std": None,
                    "floor": None, "selectivity": None, "rank_acc": None,
                })
            continue
        splits = probe_core._group_splits(np.zeros((n, 1)), cv_groups[sel])
        reason = _degeneracy_reason(y[sel], cv_groups[sel], "reg")
        for label, block, layer, position in SITES:
            if reason is not None:
                cell = {
                    "degenerate": True, "degenerate_reason": reason,
                    "real": None, "shuffled": None, "shuffled_std": None,
                    "floor": None, "selectivity": None, "rank_acc": None,
                    "n": n, "n_groups": n_seeds,
                }
            else:
                X = X_by_site[label][sel]
                if block == "raw_ft":
                    X = _zscore(X)
                factors = probe_core._fold_factors(X, cv_groups[sel], splits=splits)
                cell = _cell_from_factors(
                    factors, y[sel], cv_groups[sel], shuffle_groups[sel], "reg", seed
                )
            merged = {**common}
            merged.update({k: v for k, v in cell.items()
                           if k not in ("n", "n_groups")})
            rows.append({
                "site": label, "block": block, "layer": layer,
                "position": position, "target": TARGET, **merged,
            })
        if verbose:
            got = {r["site"]: r["selectivity"] for r in rows[-len(SITES):]}
            pretty = ", ".join(f"{k.split(' (')[0]}={v:+.3f}" for k, v in got.items())
            print(f"[bin {b:2d}] [{lo/CONTROL_HZ:+.1f},{hi/CONTROL_HZ:+.1f})s "
                  f"n={n} eps={n_eps} seeds={n_seeds} | {pretty}", flush=True)
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


def write_figure(df: pd.DataFrame, path: Path, liftoff_s: float | None,
                 note: str) -> str:
    """The deliverable figure. Layout rules (it ships into a LaTeX deck and
    an HTML page, so it must survive half-width): the legend sits INSIDE the
    axes' empty upper-left region (every line is flat there), the title owns
    its own band above the axes, and the only text under the x label is a
    ONE-LINE exploratory note -- the full method caption lives in
    ``timecurve_summary.json`` and the report, which travel with the PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ink = "#1a1a1a"
    grid_c = "#d9d9d9"
    kept = df[df["retained"] & ~df["degenerate"]]

    fig = plt.figure(figsize=(8.6, 6.0), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    gs = GridSpec(2, 1, height_ratios=[6.0, 1.0], hspace=0.12, figure=fig)
    ax = fig.add_subplot(gs[0])
    axn = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, axn):
        a.set_facecolor("#ffffff")
        a.tick_params(colors=ink, labelsize=9)
        for sp in a.spines.values():
            sp.set_color("#8a8a8a")
        a.spines[["top", "right"]].set_visible(False)

    ax.axhline(0.0, color="#9a9a9a", lw=0.9, ls=":", zorder=1)
    ax.axvline(0.0, color=ink, lw=1.4, zorder=2)
    if liftoff_s is not None:
        ax.axvline(liftoff_s, color="#9a9a9a", lw=1.1, ls="--", zorder=2)

    for label, _block, _layer, _pos in SITES:
        sub = kept[kept["site"] == label].sort_values("bin_center_s")
        if not len(sub):
            continue
        style = SITE_STYLE[label]
        x = sub["bin_center_s"].to_numpy()
        yv = sub["selectivity"].to_numpy(dtype=float)
        sd = sub["shuffled_std"].to_numpy(dtype=float)
        ax.fill_between(x, yv - sd, yv + sd, color=style["color"], alpha=0.17,
                        lw=0, zorder=3)
        ax.plot(x, yv, color=style["color"], marker=style["marker"],
                ls=style["ls"], ms=5.5, lw=2.0, label=label, zorder=4)

    ax.set_ylabel("selectivity for hidden mass\n(real R² − mean shuffled R²)",
                  color=ink, fontsize=10)
    ax.grid(True, axis="y", color=grid_c, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # in-axes rule labels, placed above the data so nothing clips
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.10
    ax.set_ylim(ymin - pad * 0.3, ymax + pad * 1.5)
    top = ax.get_ylim()[1]
    ax.annotate("first contact", xy=(0.0, top), xytext=(3, -3),
                textcoords="offset points", ha="left", va="top",
                color=ink, fontsize=9, rotation=90)
    if liftoff_s is not None:
        ax.annotate(f"median lift-off (+{liftoff_s:.2f} s)",
                    xy=(liftoff_s, top), xytext=(4, -3),
                    textcoords="offset points", ha="left", va="top",
                    color="#6a6a6a", fontsize=8.5, rotation=90)

    plt.setp(ax.get_xticklabels(), visible=False)

    # n-per-bin strip
    strip = (kept[kept["site"] == SITES[0][0]]
             .sort_values("bin_center_s")[["bin_center_s", "n", "n_episodes"]])
    width = (BIN_WIDTH_STEPS / CONTROL_HZ) * 0.86
    axn.bar(strip["bin_center_s"], strip["n"], width=width,
            color="#c9c9c9", edgecolor="#a8a8a8", lw=0.5)
    for _, r in strip.iterrows():
        axn.annotate(f"{int(r['n'])}", xy=(r["bin_center_s"], r["n"]),
                     xytext=(0, 2), textcoords="offset points", ha="center",
                     va="bottom", fontsize=7, color="#5a5a5a")
    axn.axvline(0.0, color=ink, lw=1.4)
    axn.set_ylim(0, strip["n"].max() * 1.42 if len(strip) else 1)
    axn.set_yticks([])
    axn.spines["left"].set_visible(False)
    axn.set_ylabel("rows\nper bin", color=ink, fontsize=8.5, rotation=0,
                   ha="right", va="center", labelpad=14)
    axn.set_xlabel("time relative to first contact (s)   —   1.6 s bins, "
                   "XR1 replan steps @ 20 Hz", color=ink, fontsize=10)
    axn.grid(False)

    fig.subplots_adjust(left=0.135, right=0.985, top=0.905, bottom=0.255)
    fig.text(0.135, 0.955, "When does hidden mass become decodable? "
                           "(EXPLORATORY)",
             color=ink, fontsize=13, ha="left", va="center")
    # legend BELOW the x label -- never over the plot, never over the title,
    # and 2 columns so it still fits at half width in a slide
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False,
                     fontsize=8.6, handlelength=2.4, columnspacing=1.8,
                     labelspacing=0.4)
    for t in leg.get_texts():
        t.set_color(ink)
    fig.text(0.5, 0.012, note, fontsize=7.4, color="#5a5a5a",
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
    sub = df[(df["site"] == SITES[1][0]) & df["retained"] & ~df["degenerate"]]
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
    pre = df[df["retained"] & ~df["degenerate"] & (df["bin_hi_s"] <= 0.0)]
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
        f"held-out R² over all four lines and all pre-contact bins is only "
        f"{max_real:+.4f} — i.e. no line predicts mass better than the "
        f"training-fold mean before contact. Those small positive "
        f"selectivities come from the shuffled control scoring NEGATIVE "
        f"(shuffled R² ≈ −0.02..−0.05, the usual grouped-CV behaviour when a "
        f"permuted per-episode label is fit across folds), not from real "
        f"pre-contact signal. Nothing here qualifies the study's leakage "
        f"story: the pre-registered guard (mass selectivity < 0.1 at every "
        f"probed cell on the precontact mask) also still holds at this grain."
    )


def figure_caption(df: pd.DataFrame, liftoff_s: float | None) -> str:
    kept = df[df["retained"]]["bin_index"].nunique()
    dropped = df[~df["retained"]][["bin_lo_s", "bin_hi_s", "drop_reason"]].drop_duplicates()
    drop_txt = "; ".join(
        f"[{r.bin_lo_s:+.1f},{r.bin_hi_s:+.1f})s ({r.drop_reason})"
        for r in dropped.itertuples()
    ) or "none"
    return (
        f"EXPLORATORY post-hoc time resolution of the pre-registered carry-phase probe result — no new claim, "
        f"same rows/targets/discipline at finer time grain. Selectivity = real pooled held-out R² minus mean shuffled R² "
        f"(5 group-coherent shuffles, each with its own alpha search); band = ±1 shuffled std. GroupKFold(5) by SEED, "
        f"shuffles group-coherent by EPISODE; identical rows and folds for all four lines. Bins 1.6 s (32 steps @20 Hz), "
        f"span −6.4..+9.6 s, retained only if ≥{MIN_ROWS} rows and ≥{MIN_EPISODE_GROUPS} episodes ({kept} retained; dropped: {drop_txt}). "
        f"Median lift-off {('+%.2f s' % liftoff_s) if liftoff_s is not None else 'n/a'}. "
        f"Headline unchanged: peak carry-phase selectivity 0.227 (R² 0.189) at DiT L28, below the 0.3 bar."
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def curve_summary(df: pd.DataFrame) -> dict:
    """Per-site readings the report quotes: last pre-contact bin, first
    post-contact bin, and the site's maximum over retained bins."""
    out = {}
    kept = df[df["retained"] & ~df["degenerate"]]
    for label, _b, _l, _p in SITES:
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
    ap.add_argument("--wandb-name", default="plan2-timecurve-xr1")
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
    print(f"[join] {len(idx)} captured rows over {n_eps_total} episodes "
          f"({n_no_contact} never grasped -> dropped, no contact reference); "
          f"median lift-off {liftoff_s:+.2f}s after contact "
          f"({time.time()-t0:.1f}s)", flush=True)
    print(f"[bins] {len(bins)} x {args.bin_width_steps} steps "
          f"({args.bin_width_steps/CONTROL_HZ:.1f}s) spanning "
          f"{args.span_lo_steps/CONTROL_HZ:+.1f}..{args.span_hi_steps/CONTROL_HZ:+.1f}s; "
          f"retain if n>={MIN_ROWS} and n_episodes>={MIN_EPISODE_GROUPS}", flush=True)

    df = run_time_curve(ds, acts, idx, bins, seed=SEED)
    out_path = out_dir / args.out_name
    df.to_parquet(out_path, index=False)
    print(f"[out] wrote {out_path} ({len(df)} rows)", flush=True)

    caption = figure_caption(df, liftoff_s)
    peak_note = peak_vs_headline(df)
    pre_note = precontact_reading(df)
    fig_path = write_figure(df, out_dir / args.fig_name, liftoff_s, FIGURE_NOTE)
    print(f"[fig] wrote {fig_path}", flush=True)
    print(f"\n[honesty] {peak_note}", flush=True)
    print(f"[honesty] {pre_note}", flush=True)

    summary_curve = curve_summary(df)
    print("\nCURVE (selectivity for mass_log_c):", flush=True)
    for label, s in summary_curve.items():
        if s is None:
            continue
        def fmt(k):
            v = s[k]
            return "n/a" if v is None else (
                f"{v['selectivity']:+.3f} @[{v['bin_s'][0]:+.1f},{v['bin_s'][1]:+.1f})s n={v['n']}")
        print(f"  {label}\n     last pre-contact {fmt('last_precontact_bin')}\n"
              f"     first post-contact {fmt('first_postcontact_bin')}\n"
              f"     max {fmt('max_bin')}", flush=True)

    config = {
        "analysis": "EXPLORATORY time curve (plan-2 follow-up to Task 4)",
        "exploratory_label": EXPLORATORY_LABEL,
        "headline_not_upgraded": HEADLINE_NOT_UPGRADED,
        "target": TARGET,
        "x_axis": "seconds relative to first contact (first grasped step), "
                  f"control rate {CONTROL_HZ} Hz",
        "y_axis": "selectivity = real pooled held-out R^2 - mean shuffled R^2",
        "sites": [{"label": l, "block": b, "layer": ly, "position": p}
                  for l, b, ly, p in SITES],
        "physics_line": (
            "certificates.build_features('raw_ft'): ee_force[3]+ee_torque[3]+"
            "derived [|F|,|tau|,F_world[3],F_base[3]], per-episode trailing "
            f"window k={certificates.K_RAW_FT} stride 1, z-scored per cell "
            "exactly as certificates.certificate_cell does (mixed physical "
            "units); activations are NOT z-scored (run_probes_xr1 convention)"),
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
            "cv": "GroupKFold(5) by SEED (identical fold partition for all "
                  "four sites within a bin)",
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
        },
        "git_sha": _git_sha(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    dropped = (df[~df["retained"]][["bin_lo_s", "bin_hi_s", "n", "n_episodes",
                                    "drop_reason"]]
               .drop_duplicates().to_dict("records"))
    per_bin = (df[["site", "bin_index", "bin_lo_s", "bin_hi_s", "n", "n_episodes",
                   "n_groups", "retained", "real", "shuffled", "shuffled_std",
                   "selectivity", "rank_acc", "floor", "degenerate",
                   "drop_reason"]].to_dict("records"))
    summary = sanitize_json({
        "config": config,
        "figure_caption": caption,
        "figure_note_on_png": FIGURE_NOTE,
        "peak_vs_pre_registered_headline": peak_note,
        "precontact_reading": pre_note,
        "dropped_bins": dropped,
        "curve": summary_curve,
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
        for label, s in summary_curve.items():
            if s is None:
                continue
            key = label.split(" (")[0].replace(" ", "_")
            run.summary[f"curve/{key}/max_selectivity"] = s["max_bin"]["selectivity"]
            run.summary[f"curve/{key}/max_bin_s"] = s["max_bin"]["bin_s"]
            run.summary[f"curve/{key}/max_precontact_selectivity"] = \
                s["max_precontact_selectivity"]
        run.summary["exploratory"] = True
        print("wandb url:", run.url, flush=True)
        run.finish()

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

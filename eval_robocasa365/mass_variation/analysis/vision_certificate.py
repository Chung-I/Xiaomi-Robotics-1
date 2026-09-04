# Copyright (C) 2026 Xiaomi Corporation.
"""Plan-2 amendment C: the VISION certificate (``policy_obs_vision``).

The Plan-2 certificates so far cover each policy's PROPRIOCEPTIVE format
only. Both models also see cameras, so (a) XR1's "the mass trace is
vision-borne" reading is an INFERENCE (activation probe positive + proprio
certificate null), not a measurement, and (b) pi0.5-RoboCasa's branch was
closed on partial input evidence. This module closes both by asking, per
model and in that model's own visual format: does the CAMERA channel carry
the hidden mass?

Visual format (amendment C Section 1 -- what the MODEL's format fixes)
---------------------------------------------------------------------
- ``xr1``: 4 frames at stride 2, 3 cameras, its own 0.95 centre crop.
  The frame indices come from ``window_steps``, which reproduces
  ``eval_robocasa365.entry.sample_history``'s index rule exactly (that
  function is IMPORTED as the reference in the tests, not restated).
- ``pi05_robocasa``: ONE frame (the current step), 3 cameras, its own
  ``openpi_client.image_tools.resize_with_pad``. At the stored square
  resolution that transform is provably the identity (unit-tested), so
  pi0.5's format contributes only "one frame, three cameras".

What the READER adds (disclosed, not model-side)
------------------------------------------------
Both models actually see 256x256 RGB. ``render_frames.py`` stores 96x96
GRAYSCALE per camera, and the readers below reduce further. That whole
chain -- grayscale, 96x96, the ridge reader's 512-component PCA, the CNN's
strided convolutions -- is part of the READER, never a claim about the
model's input. It is stated in every cell's ``input_channels`` and in the
report. A reader is a lower bound on channel content: it can only miss
information, never invent it, so a PASS is evidence the channel carries
mass and a FAIL is evidence only about "this reader on this reduction".

Readers (amendment C Section 2)
-------------------------------
- ``ridge_pca``: within each fold, PCA(512) fitted on the TRAINING rows
  ONLY (fitting it on all rows would leak held-out structure into the
  representation), train+test projected with that fit, then the exact
  ridge protocol every other certificate uses -- ``certificates.
  cell_from_factors``, i.e. best-alpha pooled held-out R2, per-fold R2s,
  rank_acc, 5 group-coherent shuffles with per-draw alpha search, floor.
- ``cnn``: a small seeded conv net (3 stride-2 conv blocks -> GAP ->
  linear), per fold, early-stopped on one held-out TRAIN episode, run in
  ``.venv-mibot`` through the same npz-interchange subprocess pattern the
  Plan amendment B neural readers use (``gru_worker.py --mode cnn``).
  Following that precedent, the CNN reports no shuffle control (the
  shuffle would cost 5x its training budget); its control is the
  precontact cell below, which is the amendment's own designated control.

Masks and the control (amendment C Section 3)
---------------------------------------------
Gate mask ``carry`` (R2 >= 0.3, unchanged). ``precontact`` is the CONTROL:
the three mass conditions of a seed share scene, object and placement by
the matched-seed design, so before contact a camera reader has nothing
legitimate to read. A non-null precontact cell indicates a rendering or
join artifact and VOIDS the matching carry cell -- it is reported for
every (model, reader) either way.

Run (robocasa venv; the CNN needs a free GPU):
    ~/Codes/robocasa/.venv/bin/python -m \\
        eval_robocasa365.mass_variation.analysis.vision_certificate
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from eval_robocasa365.entry import CAMERA_KEYS, center_crop
from eval_robocasa365.mass_variation.analysis import certificates, probe_core
from eval_robocasa365.mass_variation.analysis.dataset import load_study

# ----------------------------------------------------------------- constants

SEED = 0
GATE_R2 = certificates.GATE_R2          # 0.3, unchanged (amendment C Section 2)
GATE_MASK = certificates.GATE_MASK      # "carry"
CONTROL_MASK = "precontact"             # amendment C Section 3
CERT_MASKS = (GATE_MASK, CONTROL_MASK)
# Readers whose carry cell the gate reads. "sanity" (reader competence on a
# target that is manifestly in the pixels) is reported but never gated.
GATED_KINDS = ("ridge_pca", "cnn")
DEFAULT_KINDS = ("ridge_pca", "cnn", "sanity", "cnn_sanity")
CERT_NAME = "policy_obs_vision"
MODELS = ("xr1", "pi05_robocasa")
DEGENERATE_VAR_TOL = certificates.DEGENERATE_VAR_TOL

N_PCA_COMPONENTS = 512
PCA_OVERSAMPLE = 10       # randomized range-finder oversampling
PCA_N_ITER = 4            # power iterations (sklearn's randomized_svd default)

# Ridge alpha grid for THIS reader only (review fix 1). ``probe_core.ALPHAS``
# (1e-2..1e4) is the study's registered grid, calibrated for the
# proprioceptive path whose features are globally z-SCORED. This reader's
# features are UNSCALED principal-component scores whose leading singular
# value is ~2700-2970 (s^2 ~ 8e6), so every alpha in the registered grid is
# negligible against s^2: the grid collapsed to essentially OLS on 512 PCs
# and the selected alpha sat at the grid's UPPER EDGE in every cell -- a
# TRUNCATED search, not a converged one.
#
# Fix chosen: WIDEN THE GRID rather than z-score the PC scores. Rescaling the
# scores would change the estimator itself (whitened-PCA ridge weights the
# trailing, noise-dominated directions like the leading ones); widening
# changes only the search range, which is what was actually broken. The real
# fit and all five shuffle draws search the identical widened grid, so the
# control stays fair.
VISION_ALPHAS = 10.0 ** np.arange(-2, 13)   # 1e-2 .. 1e12
# Ridge's alpha -> infinity limit IS the predict-the-training-fold-mean floor,
# so a top-edge selection is only TRUNCATED if its score still differs from
# that floor. Within this tolerance the grid has reached the mean predictor
# and a top-edge selection is the converged answer "no signal above the
# floor", not a cut-off search.
ALPHA_SATURATION_TOL = 1e-3

XR1_CROP_RATIO = 0.95     # entry.py --crop-ratio default
STORE_SIDE_PX = 96        # render_frames.STORE_SIDE_PX (reader-side)

# Each model's own visual format: how many frames, at what stride, and the
# model's own per-frame geometric transform (see the module docstring).
VISUAL_FORMATS = {
    # entry.py --obs-history 4 / --obs-interval 2 defaults
    "xr1": {"window": {"k": 4, "stride": 2}, "transform": "center_crop_0.95"},
    # pi05_client.pack_pi05_element takes the LAST frame only
    "pi05_robocasa": {"window": {"k": 1, "stride": 1}, "transform": "resize_with_pad"},
}

# CNN reader budget (amendment C Section 2's "fair budget, early stop")
CNN_MAX_EPOCHS = 200
CNN_PATIENCE = 15
CNN_BATCH = 128
CNN_BUDGET_S = 300.0      # <= 5 min per (model, mask) cell, all folds together
CNN_RELIABILITY_NOTE = (
    "trained reader in a data-starved regime (35 seed groups, 3 mass levels): "
    "treat as a noisy reader, not a tight bound on channel content; no shuffle "
    "control (see the module docstring) -- its control is the precontact cell"
)

DEFAULT_PHASE1_ROOT = "output/mass_variation/phase1"
DEFAULT_POLICY_STATE_ROOT = "output/mass_variation/policy_state"
DEFAULT_FRAMES_ROOT = "output/mass_variation/frames"
DEFAULT_OUT = "output/mass_variation/analysis/vision_certificate.json"


def input_channels_for(model: str, kind: str) -> list[str]:
    """The cell's honest input disclosure -- model-side format vs
    reader-side reduction, kept apart on purpose."""
    fmt = VISUAL_FORMATS[model]
    k, stride = fmt["window"]["k"], fmt["window"]["stride"]
    reader = {
        "ridge_pca": (f"READER: 96x96 grayscale per camera (storage), flattened "
                      f"{k}x{len(CAMERA_KEYS)}x{STORE_SIDE_PX}x{STORE_SIDE_PX} = "
                      f"{k * len(CAMERA_KEYS) * STORE_SIDE_PX ** 2} dims, then "
                      f"PCA({N_PCA_COMPONENTS}) FITTED WITHIN EACH TRAIN FOLD, "
                      f"then ridge (same alpha grid / shuffles / folds as every "
                      f"other certificate)"),
        "cnn": (f"READER: 96x96 grayscale per camera (storage) as "
                f"{k * len(CAMERA_KEYS)} input channels; 3 stride-2 conv blocks "
                f"-> GAP -> linear, seeded, early-stopped on a held-out train "
                f"episode"),
    }[kind]
    return [
        (f"MODEL FORMAT ({model}): {k} frame(s) at stride {stride} x "
         f"{len(CAMERA_KEYS)} cameras {list(CAMERA_KEYS)}, per-frame "
         f"{fmt['transform']}"),
        reader,
        ("NOTE: both models actually see 256x256 RGB; the 96x96 grayscale "
         "storage and every reduction above are READER-side, not the model's "
         "input format (amendment C Section 1)"),
    ]


# ------------------------------------------------------- pure (unit-tested)


def window_steps(t: int, k: int, stride: int) -> list[int]:
    """The absolute step indices a policy at step ``t`` reads, oldest first.

    ``eval_robocasa365.entry.sample_history``'s index rule
    (``max(0, len - 1 - (k - 1 - i) * interval)`` over a deque of maxlen
    ``(k - 1) * stride + 1``), expressed on absolute step indices:
    ``max(0, t - (k - 1 - i) * stride)``. The test suite pins this against
    the imported ``sample_history`` driven with a real deque, so the two
    can never drift.
    """
    if k < 1 or stride < 1:
        raise ValueError(f"window_steps: k and stride must be >= 1, got {k}/{stride}")
    return [max(0, t - (k - 1 - i) * stride) for i in range(k)]


def apply_format_transform(frame: np.ndarray, model: str) -> np.ndarray:
    """One stored 96x96 grayscale camera frame through THAT MODEL's own
    per-frame image transform.

    - ``xr1``: ``entry.center_crop(frame, 0.95)`` -- imported, not restated.
    - ``pi05_robocasa``: ``openpi_client.image_tools.resize_with_pad`` to the
      same square side, which is the identity here (unit-tested); pi0.5's
      live pipeline pads/resizes 256->224, and at a square 96x96 input that
      is a no-op.

    Applying the crop at the STORED resolution rather than at 256 is a
    reader-side approximation of the live order (crop-then-downsample), and
    is disclosed as such.
    """
    array = np.asarray(frame, dtype=np.uint8)
    if model == "xr1":
        return np.asarray(center_crop(array, XR1_CROP_RATIO), dtype=np.uint8)
    if model == "pi05_robocasa":
        from openpi_client import image_tools
        side = array.shape[0]
        return np.asarray(
            image_tools.resize_with_pad(array[..., None], side, side), dtype=np.uint8
        )[..., 0]
    raise ValueError(f"apply_format_transform: unknown model {model!r}")


def gather_visual_rows(frames, steps, row_steps, k: int, stride: int) -> np.ndarray:
    """``(n, k, C, H, W)`` uint8: for each row step in ``row_steps``, that
    row's ``k``-frame stride-``stride`` window gathered out of one episode's
    stored ``frames`` (indexed by the stored ``steps``).

    Raises ``KeyError`` if a needed step was not stored -- silently
    substituting a neighbour would corrupt the format.
    """
    frames = np.asarray(frames)
    steps = np.asarray(steps).astype(np.int64)
    lookup = {int(s): i for i, s in enumerate(steps)}
    rows = np.asarray(row_steps).astype(np.int64)
    idx = np.empty((len(rows), k), dtype=np.int64)
    for r, t in enumerate(rows):
        for j, want in enumerate(window_steps(int(t), k, stride)):
            try:
                idx[r, j] = lookup[want]
            except KeyError as exc:
                raise KeyError(
                    f"gather_visual_rows: step {want} (needed by row step {t}) "
                    f"is not in the stored keep-set"
                ) from exc
    return frames[idx]


def pca_fit(X, n_components: int, seed: int = SEED,
            n_iter: int = PCA_N_ITER, oversample: int = PCA_OVERSAMPLE):
    """Randomized PCA of ``X`` ``(n, D)`` -> ``(mean (D,), components
    (k, D))``, ``k = min(n_components, n, D)``, components orthonormal and
    ordered by explained variance.

    A randomized range finder (Halko et al.), not ``sklearn.PCA``, for one
    reason: the XR1 design matrix is ``(~13k, 110592)`` float32 and an exact
    SVD -- or sklearn's copy-on-center -- would need tens of GB. Every
    matmul here is done against ``X`` with the mean subtracted ON THE FLY
    (``(X - mu) @ Q == X @ Q - mu @ Q``), so no centered copy of ``X`` is
    ever materialised. Seeded, so the fit is reproducible.
    """
    X = np.asarray(X, dtype=np.float32)
    n, D = X.shape
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    k = int(min(n_components, n, D))
    if k < 1:
        raise ValueError(f"pca_fit: no components available for shape {X.shape}")
    ell = int(min(k + oversample, n, D))

    def cmm(Q):            # (X - mean) @ Q
        return X @ Q - mean @ Q

    def cmm_T(Y):          # (X - mean).T @ Y
        return X.T @ Y - mean[:, None] * Y.sum(axis=0, keepdims=True)

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(cmm(rng.standard_normal((D, ell)).astype(np.float32)))
    for _ in range(n_iter):
        Z, _ = np.linalg.qr(cmm_T(Q))
        Q, _ = np.linalg.qr(cmm(Z))
    B = cmm_T(Q).T                       # (ell, D) = Q^T (X - mean)
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return mean, np.ascontiguousarray(Vt[:k], dtype=np.float32)


def pca_transform(X, mean, components) -> np.ndarray:
    """Project ``X`` onto a ``pca_fit`` result: ``(X - mean) @ components.T``,
    computed without materialising the centered copy."""
    X = np.asarray(X, dtype=np.float32)
    components = np.asarray(components, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    return X @ components.T - (mean @ components.T)[None, :]


def pca_fold_factors(X, cv_groups, n_components: int = N_PCA_COMPONENTS,
                     seed: int = SEED, splits=None) -> list[dict]:
    """``probe_core._fold_factors``-shaped factors with a PCA fitted INSIDE
    each fold.

    Per fold: fit ``pca_fit`` on ``X[tr]`` ONLY (amendment C Section 2's
    leakage rule -- an all-rows PCA would carry held-out structure into the
    representation every fold is then scored on), project train and test
    with that fit, and hand the resulting ``(n, k)`` scores to the same
    centered-SVD ridge factorisation ``probe_core`` uses everywhere else.
    Each returned dict additionally carries ``pca_mean``/``pca_components``
    so the in-fold provenance is directly assertable (it is, in
    ``test_vision_certificate.py``).

    PCA scores are NOT rescaled: ridge on variance-ordered principal
    components (principal-component ridge regression) is the standard form,
    and rescaling them to unit variance would hand the trailing, noise-
    dominated directions the same weight as the leading ones.
    """
    X = np.asarray(X, dtype=np.float32)
    cv_groups = np.asarray(cv_groups)
    if splits is None:
        splits = probe_core._group_splits(np.zeros((X.shape[0], 1)), cv_groups)
    factors = []
    for tr, te in splits:
        mean, comps = pca_fit(X[tr], n_components, seed=seed)
        Z_tr = pca_transform(X[tr], mean, comps).astype(np.float64)
        Z_te = pca_transform(X[te], mean, comps).astype(np.float64)
        mu = Z_tr.mean(axis=0)
        U, s, Vt = np.linalg.svd(Z_tr - mu, full_matrices=False)
        factors.append({
            "tr": tr, "te": te, "U": U, "s": s, "G": (Z_te - mu) @ Vt.T,
            "pca_mean": mean, "pca_components": comps,
        })
    return factors


class AlphaGridTruncated(RuntimeError):
    """The ridge alpha search selected a value at an END of its grid and that
    end is NOT the mean-predictor limit -- i.e. the reported score is a
    property of where the grid stopped, not of the data. Never widen the
    tolerance to silence this; widen the grid."""


@contextlib.contextmanager
def alpha_grid(alphas):
    """Run ``probe_core``'s ridge path against ``alphas`` instead of its
    module-level registered grid.

    ``probe_core.py`` is COPIED VERBATIM from the sibling study (its header
    note forbids editing it), and it reads ``ALPHAS`` at module scope, so a
    scoped, restoring rebind is how this reader supplies its own grid without
    touching that file or duplicating the statistic. Everything inside the
    block -- the real fit AND every shuffle draw -- sees the same grid.
    """
    previous = probe_core.ALPHAS
    probe_core.ALPHAS = np.asarray(alphas, dtype=float)
    try:
        yield
    finally:
        probe_core.ALPHAS = previous


def alpha_curve(factors, y, alphas) -> list[tuple[float, float]]:
    """``[(alpha, pooled held-out R2)]`` -- the search's own scoring code
    driven one alpha at a time (single-element grids), so the curve cannot
    drift from what the selection actually optimises."""
    y = np.asarray(y, dtype=np.float64)
    curve = []
    for a in alphas:
        with alpha_grid([a]):
            curve.append((float(a),
                          float(probe_core._cv_pooled_best_factored(factors, y, "reg"))))
    return curve


def check_alpha_boundary(curve, floor: float,
                         tol: float = ALPHA_SATURATION_TOL) -> dict:
    """Review fix 1's boundary-saturation guard.

    Raises :class:`AlphaGridTruncated` if the best alpha is the grid's LOWEST
    (always a truncation -- a weaker penalty was never tried) or its HIGHEST
    while still scoring materially above the mean-predictor ``floor`` (the
    alpha -> infinity limit, so the grid stopped before the penalty could
    finish taking effect). A top-edge selection whose score has already
    reached the floor is accepted and flagged ``saturated_at_floor``: the
    search converged, and its answer is "nothing beats predicting the mean".
    """
    alphas = [a for a, _ in curve]
    scores = [s for _, s in curve]
    best = int(np.argmax(scores))
    at_floor = bool(abs(scores[best] - floor) <= tol)
    info = {
        "alpha_selected": alphas[best],
        "alpha_grid_min": alphas[0], "alpha_grid_max": alphas[-1],
        "alpha_at_lower_bound": best == 0,
        "alpha_at_upper_bound": best == len(alphas) - 1,
        "alpha_saturated_at_floor": at_floor,
        "alpha_curve": [[a, s] for a, s in curve],
    }
    if best == 0:
        raise AlphaGridTruncated(
            f"best alpha {alphas[best]:g} is the grid's LOWEST -- the search "
            f"never tried a weaker penalty; widen the grid downward")
    if best == len(alphas) - 1 and not at_floor:
        raise AlphaGridTruncated(
            f"best alpha {alphas[best]:g} is the grid's HIGHEST and its score "
            f"{scores[best]:.6f} is still {scores[best] - floor:+.6f} from the "
            f"mean-predictor floor {floor:.6f} -- the search was truncated; "
            f"widen the grid upward")
    return info


def vision_certificate_cell(X, y, cv_groups, shuffle_groups,
                            n_components: int = N_PCA_COMPONENTS,
                            seed: int = SEED, alphas=VISION_ALPHAS) -> dict:
    """One ``ridge_pca`` cell: in-fold PCA, then the EXACT certificate
    statistic every other certificate uses
    (``certificates.cell_from_factors``), searched over ``alphas`` (see
    :data:`VISION_ALPHAS` for why this reader needs its own grid).
    Degenerate guards mirror ``certificates.certificate_cell``; the alpha
    boundary guard raises rather than reporting a truncated number."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return certificates.degenerate_cell("empty mask")
    if float(np.var(y)) < DEGENERATE_VAR_TOL:
        return certificates.degenerate_cell(
            f"target variance {float(np.var(y)):.3e} < {DEGENERATE_VAR_TOL}")
    cv_groups = np.asarray(cv_groups)
    splits = probe_core._group_splits(np.zeros((len(y), 1)), cv_groups)
    factors = pca_fold_factors(X, cv_groups, n_components=n_components,
                               seed=seed, splits=splits)
    with alpha_grid(alphas):
        cell = certificates.cell_from_factors(
            factors, splits, y, cv_groups, np.asarray(shuffle_groups), seed=seed)
    floor = probe_core._floor(y, cv_groups, "reg", splits=splits)
    cell.update(check_alpha_boundary(alpha_curve(factors, y, alphas), floor))
    cell["pca_components_used"] = int(factors[0]["pca_components"].shape[0])
    return cell


# --------------------------------------------------------- design matrices


def build_visual_matrix(d: dict, model: str, mask_name: str, frames_root: Path,
                        row_stride: int = 1, verbose: bool = True):
    """``(X_uint8 (n, k*C, H, W), rows (n,))`` for one (model, mask): the
    model's own visual format assembled from the stored frame npz, over the
    masked rows of ``load_study``'s row table.

    ``rows`` indexes back into ``d``'s arrays (labels, seeds, episode ids).
    ``row_stride`` sub-samples rows WITHIN each episode (never dropping an
    episode) when a reader's budget demands it -- the rule and rate are
    recorded in the output JSON.
    """
    fmt = VISUAL_FORMATS[model]
    k, stride = fmt["window"]["k"], fmt["window"]["stride"]
    mask = d["masks"][mask_name]
    eid = d["episode_id"]
    step = d["step"]

    blocks, row_idx = [], []
    for episode in dict.fromkeys(eid.tolist()):           # study row order
        in_ep = eid == episode
        sel = np.flatnonzero(in_ep & mask)
        if sel.size == 0:
            continue
        if row_stride > 1:
            sel = sel[::row_stride]
        condition, ep_name = episode.split("/")
        seed = int(ep_name.split("_")[1])
        path = Path(frames_root) / model / condition / f"ep_{seed}.npz"
        with np.load(path) as z:
            frames, steps = z["frames"], z["steps"]
        window = gather_visual_rows(frames, steps, step[sel], k=k, stride=stride)
        # model's own per-frame transform, then flatten (frame, camera) into
        # the reader's channel axis
        n, kk, C, H, W = window.shape
        out = np.empty((n, kk * C, H, W), dtype=np.uint8)
        for i in range(n):
            for j in range(kk):
                for c in range(C):
                    out[i, j * C + c] = apply_format_transform(window[i, j, c], model)
        blocks.append(out)
        row_idx.append(sel)
    if not blocks:
        return (np.zeros((0, k * len(CAMERA_KEYS), STORE_SIDE_PX, STORE_SIDE_PX),
                         dtype=np.uint8), np.zeros(0, dtype=np.int64))
    X = np.concatenate(blocks, axis=0)
    rows = np.concatenate(row_idx)
    if verbose:
        print(f"[build] {model}/{mask_name}: X={X.shape} uint8 "
              f"({X.nbytes / 1e9:.2f} GB) rows={len(rows)} "
              f"episodes={len(blocks)} row_stride={row_stride}", flush=True)
    return X, rows


# --------------------------------------------------------------- CNN reader


def run_cnn_certificate(X_uint8, y, seeds_rows, cnn_python: str, device: str,
                        tmp_dir: Path, max_epochs: int = CNN_MAX_EPOCHS,
                        budget_s: float = CNN_BUDGET_S,
                        patience: int = CNN_PATIENCE,
                        batch: int = CNN_BATCH) -> dict:
    """One CNN cell, dispatched to ``.venv-mibot`` exactly the way
    ``certificates.run_neural_certificate`` dispatches the MLP/GRU readers
    (npz interchange, folds computed HERE so the CNN scores byte-identical
    seed-grouped folds to the ridge path, pooled R2/rank_acc computed HERE
    so the metric implementations stay single-sourced).

    Interchange uses ``np.savez`` (uncompressed): the image payload is up to
    ~1.5 GB of uint8 and zlib on it would cost more than the training.
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return certificates.degenerate_cell("empty mask")
    if float(np.var(y)) < DEGENERATE_VAR_TOL:
        return certificates.degenerate_cell("constant target under mask")

    seeds_rows = np.asarray(seeds_rows, dtype=np.int64)
    splits = probe_core._group_splits(np.zeros((len(y), 1)), seeds_rows)
    folds = [(sorted(set(seeds_rows[tr].tolist())), sorted(set(seeds_rows[te].tolist())))
             for tr, te in splits]

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_npz, out_npz = tmp_dir / "cnn_in.npz", tmp_dir / "cnn_out.npz"
    try:
        np.savez(in_npz, X=np.asarray(X_uint8, dtype=np.uint8), y=y,
                 mask=np.ones(len(y), dtype=bool), seed=seeds_rows,
                 folds_json=np.array(json.dumps(folds)))
        cmd = [cnn_python, "-u", "-m",
               "eval_robocasa365.mass_variation.analysis.gru_worker",
               str(in_npz), str(out_npz), "--mode", "cnn", "--device", device,
               "--max-epochs", str(max_epochs), "--budget-s", str(budget_s),
               "--patience", str(patience), "--row-batch", str(batch)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parents[3]))
        if proc.returncode != 0:
            raise RuntimeError(
                f"cnn worker failed (rc={proc.returncode}):\n"
                f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")
        sys.stdout.write(proc.stdout)
        with np.load(out_npz) as z:
            res = {k: z[k] for k in z.files}
    finally:
        in_npz.unlink(missing_ok=True)
        out_npz.unlink(missing_ok=True)

    yt, yp = res["y_true"], res["y_pred"]
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return {
        "degenerate": False,
        "r2_pooled": (1.0 - float(((yp - yt) ** 2).sum()) / ss_tot
                      if ss_tot > 0 else float("nan")),
        "r2_folds": [float(r) for r in res["r2_folds"]],
        "rank_acc": float(certificates.rank_acc_levels(yt, yp)),
        "shuffled": None,       # see the module docstring: control is precontact
        "shuffled_std": None,
        "selectivity": None,
        "floor": None,
        "n": int(len(y)),
        "n_groups": int(len(np.unique(seeds_rows))),
        "epochs_per_fold": [int(e) for e in res["epochs_per_fold"]],
        "budget_hit": bool(res["budget_hit"]),
        "wall_s": float(res["wall_s"]),
        "reliability_note": CNN_RELIABILITY_NOTE,
    }


# ----------------------------------------------------------------------- main


def gate_verdict(by_kind: dict) -> dict:
    """PASS/FAIL on the unchanged pre-registered gate (carry R2 >= 0.3),
    plus the amendment C Section 3 control read: a non-null precontact cell
    VOIDS the carry cell for that reader."""
    entry = {"gate_r2": GATE_R2, "gate_mask": GATE_MASK,
             "control_mask": CONTROL_MASK, "control_null_r2_max": 0.0}
    passes, voided = [], []
    for kind, by_mask in by_kind.items():
        carry = by_mask.get(GATE_MASK)
        control = by_mask.get(CONTROL_MASK)
        r2 = None if carry is None or carry.get("degenerate") else carry["r2_pooled"]
        c_r2 = None if control is None or control.get("degenerate") else control["r2_pooled"]
        # "null" == the control reader does no better than predicting the
        # training mean, i.e. R2 <= 0. A positive control R2 means the
        # pre-contact frames already separate the mass conditions, which the
        # matched-seed design says is impossible -> artifact.
        # A MISSING control is "not yet measured", not "not null" -- only a
        # measured, positive control voids the carry cell.
        is_null = None if c_r2 is None else bool(c_r2 <= 0.0)
        ok = bool(r2 is not None and r2 >= GATE_R2)
        entry[f"{kind}_r2"] = r2
        entry[f"{kind}_control_r2"] = c_r2
        entry[f"{kind}_control_null"] = is_null
        entry[f"{kind}_pass"] = bool(ok and is_null)
        entry[f"{kind}_voided"] = bool(is_null is False)
        passes.append(bool(ok and is_null))
        voided.append(is_null is False)
    entry["pass"] = any(passes)
    entry["any_voided"] = any(voided)
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1-root", default=DEFAULT_PHASE1_ROOT)
    ap.add_argument("--policy-state-root", default=DEFAULT_POLICY_STATE_ROOT)
    ap.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--masks", nargs="+", default=list(CERT_MASKS))
    ap.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS))
    ap.add_argument("--n-components", type=int, default=N_PCA_COMPONENTS)
    ap.add_argument("--recompute-kinds", nargs="*", default=[],
                    help="Discard any cached cells of these kinds and "
                         "recompute them (e.g. after a reader change).")
    ap.add_argument("--row-stride", type=int, default=1,
                    help="Sub-sample rows WITHIN episodes (never drop an "
                         "episode) if a reader's budget demands it.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cnn-python", default=".venv-mibot/bin/python")
    ap.add_argument("--tmp-dir", default="output/mass_variation/analysis/tmp")
    ap.add_argument("--max-epochs", type=int, default=CNN_MAX_EPOCHS)
    ap.add_argument("--budget-s", type=float, default=CNN_BUDGET_S)
    ap.add_argument("--patience", type=int, default=CNN_PATIENCE)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="plan2-vision-certificate")
    args = ap.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    if not Path(args.cnn_python).is_absolute():
        args.cnn_python = str(repo_root / args.cnn_python)

    # Resumable, so each (model, mask) can be driven in its own bounded
    # foreground invocation: cells already in --out are reused verbatim and
    # never recomputed; the gate table and JSON are rebuilt every run.
    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            previous = None
        if previous:
            done = {(c["model"], c["kind"], c["mask"]): c
                    for c in previous.get("cells", [])
                    if c["kind"] not in set(args.recompute_kinds)}
            print(f"[resume] {len(done)} cell(s) already in {out_path}"
                  + (f"; recomputing kind(s) {args.recompute_kinds}"
                     if args.recompute_kinds else ""), flush=True)

    cells, gates = [], {}
    for model in args.models:
        d = load_study(model, args.phase1_root, args.policy_state_root)
        print(f"[load] {model}: N={len(d['step'])} "
              f"masks={{{', '.join(f'{k}:{int(v.sum())}' for k, v in d['masks'].items())}}}",
              flush=True)
        by_kind = {kind: {} for kind in args.kinds}
        for mask_name in args.masks:
            for kind in args.kinds:
                cached = done.get((model, kind, mask_name))
                if cached is not None:
                    by_kind[kind][mask_name] = cached
                    cells.append(cached)
                    print(f"[cached] {model}/{kind}/{mask_name}: "
                          f"R2={cached['r2_pooled']}", flush=True)
            todo = [k for k in args.kinds if mask_name not in by_kind[k]]
            if not todo:
                continue

            X, rows = build_visual_matrix(
                d, model, mask_name, Path(args.frames_root), row_stride=args.row_stride)
            y = d["mass_log_c"][rows]
            seeds_rows = d["seed"][rows]
            episodes_rows = d["episode_id"][rows]

            if "ridge_pca" in todo:
                t0 = time.time()
                flat = X.reshape(X.shape[0], -1).astype(np.float32)
                flat /= 255.0
                cell = vision_certificate_cell(
                    flat, y, seeds_rows, episodes_rows,
                    n_components=args.n_components, seed=SEED)
                del flat
                cell.update(model=model, certificate=CERT_NAME, kind="ridge_pca",
                            mask=mask_name, input_channels=input_channels_for(model, "ridge_pca"),
                            row_stride=args.row_stride,
                            wall_s=round(time.time() - t0, 1))
                by_kind["ridge_pca"][mask_name] = cell
                cells.append(cell)
                print(f"[ridge_pca] {model}/{mask_name}: R2={cell['r2_pooled']} "
                      f"folds={cell['r2_folds']} sel={cell['selectivity']} "
                      f"rank_acc={cell['rank_acc']} n={cell['n']} "
                      f"({cell['wall_s']}s)", flush=True)

            if "sanity" in todo:
                # READER-COMPETENCE control (not a certificate, not gated):
                # the same ridge_pca reader on a target that is manifestly
                # IN these pixels -- the carried object's world height. A
                # mass FAIL is only interpretable if the reader can read
                # SOMETHING off the frames it was given; a near-zero R2 here
                # would mean the pipeline (join, render, format) is broken,
                # not that the camera lacks mass.
                t0 = time.time()
                flat = X.reshape(X.shape[0], -1).astype(np.float32)
                flat /= 255.0
                cell = vision_certificate_cell(
                    flat, d["obj_pos"][rows, 2], seeds_rows, episodes_rows,
                    n_components=args.n_components, seed=SEED)
                del flat
                cell.update(model=model, certificate="reader_competence_obj_z",
                            kind="sanity", mask=mask_name,
                            input_channels=input_channels_for(model, "ridge_pca"),
                            row_stride=args.row_stride,
                            target="obj_pos_z (object world height, metres)",
                            wall_s=round(time.time() - t0, 1))
                by_kind["sanity"][mask_name] = cell
                cells.append(cell)
                print(f"[sanity] {model}/{mask_name} obj_z: R2={cell['r2_pooled']} "
                      f"folds={cell['r2_folds']} n={cell['n']} "
                      f"({cell['wall_s']}s)", flush=True)

            if "cnn_sanity" in todo:
                # The CNN's own reader-competence control, same target.
                cell = run_cnn_certificate(
                    X, d["obj_pos"][rows, 2], seeds_rows, args.cnn_python, args.device,
                    repo_root / args.tmp_dir, max_epochs=args.max_epochs,
                    budget_s=args.budget_s, patience=args.patience)
                cell.update(model=model, certificate="reader_competence_obj_z",
                            kind="cnn_sanity", mask=mask_name,
                            input_channels=input_channels_for(model, "cnn"),
                            row_stride=args.row_stride,
                            target="obj_pos_z (object world height, metres)")
                by_kind["cnn_sanity"][mask_name] = cell
                cells.append(cell)
                print(f"[cnn_sanity] {model}/{mask_name} obj_z: R2={cell['r2_pooled']} "
                      f"folds={cell.get('r2_folds')} n={cell['n']}", flush=True)

            if "cnn" in todo:
                cell = run_cnn_certificate(
                    X, y, seeds_rows, args.cnn_python, args.device,
                    repo_root / args.tmp_dir, max_epochs=args.max_epochs,
                    budget_s=args.budget_s, patience=args.patience)
                cell.update(model=model, certificate=CERT_NAME, kind="cnn",
                            mask=mask_name, input_channels=input_channels_for(model, "cnn"),
                            row_stride=args.row_stride)
                by_kind["cnn"][mask_name] = cell
                cells.append(cell)
                print(f"[cnn] {model}/{mask_name}: R2={cell['r2_pooled']} "
                      f"folds={cell.get('r2_folds')} rank_acc={cell.get('rank_acc')} "
                      f"epochs={cell.get('epochs_per_fold')} n={cell['n']} "
                      f"({cell.get('wall_s')}s)", flush=True)
            del X

    # Carry forward every cached cell this invocation did not visit, so a
    # partial run (e.g. one model) never drops the other's results, then
    # rebuild the gate table from the FULL merged cell set.
    visited = {(c["model"], c["kind"], c["mask"]) for c in cells}
    for key, cell in done.items():
        if key not in visited:
            cells.append(cell)
    cells.sort(key=lambda c: (c["model"], c["kind"], c["mask"]))

    # "sanity" is a reader-competence control, never a certificate reader:
    # it is reported alongside but takes no part in the gate verdict.
    all_kinds = [k for k in sorted({c["kind"] for c in cells}) if k in GATED_KINDS]
    for model in sorted({c["model"] for c in cells}):
        by_kind = {kind: {c["mask"]: c for c in cells
                          if c["model"] == model and c["kind"] == kind}
                   for kind in all_kinds}
        gates[f"{model}/{CERT_NAME}"] = gate_verdict(
            {k: v for k, v in by_kind.items() if v})

    print(f"\n===== AMENDMENT C VISION CERTIFICATE GATE TABLE "
          f"(mass R2 >= {GATE_R2} on '{GATE_MASK}', '{CONTROL_MASK}' must be null) =====",
          flush=True)
    for key, g in gates.items():
        for kind in all_kinds:
            if f"{kind}_r2" not in g:
                continue
            r2, c_r2 = g.get(f"{kind}_r2"), g.get(f"{kind}_control_r2")
            print(f"  {key:34s} {kind:10s} carry_R2="
                  f"{'None' if r2 is None else f'{r2:+.4f}':>9s}  "
                  f"precontact_R2={'None' if c_r2 is None else f'{c_r2:+.4f}':>9s}  "
                  f"-> {'PASS' if g.get(f'{kind}_pass') else 'FAIL'}"
                  f"{'  [VOIDED: control not null]' if g.get(f'{kind}_voided') else ''}",
                  flush=True)
        print(f"  {key:34s} {'OVERALL':10s} -> {'PASS' if g['pass'] else 'FAIL'}", flush=True)
    print("=" * 96, flush=True)

    config = {
        "amendment": ("C (2026-09-04): vision certificate -- each model's own "
                      "visual format, frames re-rendered from the bit-exact "
                      "replays; readers = in-fold-PCA ridge + small seeded CNN; "
                      "gate unchanged (carry R2 >= 0.3); precontact is the "
                      "control and a non-null control voids the carry cell"),
        "seed": SEED, "gate_r2": GATE_R2, "gate_mask": GATE_MASK,
        "control_mask": CONTROL_MASK, "cert_masks": list(args.masks),
        "certificate": CERT_NAME,
        "visual_formats": VISUAL_FORMATS,
        "camera_keys": list(CAMERA_KEYS),
        "storage": {
            "side_px": STORE_SIDE_PX, "channels": "grayscale",
            "source_px": 256, "source_channels": "RGB",
            "disclosure": ("the 96x96 grayscale storage and every further "
                           "reduction (PCA / strided conv) are part of the "
                           "READER, not any model's input format"),
        },
        "pca": {"n_components": args.n_components, "n_iter": PCA_N_ITER,
                "oversample": PCA_OVERSAMPLE,
                "fit_scope": "TRAIN ROWS OF EACH FOLD ONLY (leakage rule)",
                "scores_rescaled": False},
        "cnn": {"max_epochs": args.max_epochs, "patience": args.patience,
                "batch": CNN_BATCH, "budget_s": args.budget_s,
                "arch": "3x(conv3x3 stride2 + BN + ReLU) -> GAP -> linear",
                "runner": f"gru_worker.py --mode cnn via {args.cnn_python}",
                "shuffle_control": None,
                "reliability_note": CNN_RELIABILITY_NOTE},
        "row_subsample": {"rule": "every row_stride-th masked row WITHIN each "
                                  "episode (episodes are never dropped)",
                          "row_stride": args.row_stride,
                          "rate": 1.0 / args.row_stride},
        "alphas": VISION_ALPHAS.tolist(),
        "alphas_registered_proprio_path": probe_core.ALPHAS.tolist(),
        "alpha_grid_note": (
            "review fix 1: the registered 1e-2..1e4 grid is calibrated for the "
            "z-scored proprioceptive features; on UNSCALED PCA scores "
            "(s[0]~2700-2970) every one of its alphas is negligible against "
            "s^2, so the search saturated at the grid's upper edge in every "
            "cell (effectively OLS on 512 PCs). Fixed by WIDENING the grid to "
            "1e12 -- chosen over z-scoring the PC scores because widening "
            "changes only the search range, while rescaling would change the "
            "estimator (whitened-PCA ridge). Real fit and all 5 shuffle draws "
            "share the widened grid. check_alpha_boundary raises if a "
            "selection sits at either end without having reached the "
            "mean-predictor floor."),
        "n_splits": probe_core.N_SPLITS,
        "n_shuffles": probe_core.N_SHUFFLES,
        "cv_groups": "seed (35 matched-pair groups)",
        "shuffle_groups": "episode (condition/seed) -- certificates.py's hybrid rule",
        "models": list(args.models),
        "phase1_root": args.phase1_root, "policy_state_root": args.policy_state_root,
        "frames_root": args.frames_root,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    try:
        import sklearn
        config["versions"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    if "cnn" in args.kinds:
        proc = subprocess.run([args.cnn_python, "-c", "import torch; print(torch.__version__)"],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            config["versions"]["cnn_torch"] = proc.stdout.strip()

    # "complete" == the CERTIFICATE proper (both gated readers x both masks x
    # both models) is present. The reader-competence cells are reported
    # separately and are defined only on `carry`, where the object visibly
    # moves -- pre-contact its height is near-constant, so a competence cell
    # there would measure nothing.
    have = {(c["model"], c["kind"], c["mask"]) for c in cells}
    config["complete"] = all(
        (model, kind, mask) in have
        for model in MODELS for kind in GATED_KINDS for mask in CERT_MASKS
    )
    config["reader_competence_cells"] = sorted(
        f"{m}/{k}/{mk}" for (m, k, mk) in have if k in ("sanity", "cnn_sanity"))

    out = certificates.sanitize_json({"config": config, "gates": gates, "cells": cells})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_path}", flush=True)

    if not args.no_wandb:
        import pandas as pd
        import wandb
        run = wandb.init(project="mass-com-xr1", job_type="analysis",
                         name=args.wandb_name, config=config)
        rows = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                 for k, v in c.items()} for c in cells]
        run.log({"vision_certificate": wandb.Table(dataframe=pd.DataFrame(rows))})
        run.summary.update({f"gate/{key}/{k}": v for key, g in gates.items()
                            for k, v in g.items() if v is not None})
        print("wandb url:", run.url, flush=True)
        run.finish()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

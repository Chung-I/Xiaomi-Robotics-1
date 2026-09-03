# Copyright (C) 2026 Xiaomi Corporation.
"""Torch-only neural-reader worker for the Plan-2 Task 2 certificates.

Runs in ``.venv-mibot`` (torch 2.8.0+cu128 -- the robocasa venv's torch
2.7.1+cu126 has no sm_120 kernels for the RTX 5090), invoked as a
subprocess by ``certificates.py`` with npz interchange, per the plan's
two-venv rule. Imports ONLY numpy + torch (no sklearn/wandb here): the
seed-grouped fold partition is computed by the caller (probe_core /
sklearn side) and shipped inside the exchange npz.

Two reader modes (Plan amendment B Section 2 -- fair training budgets):

- ``--mode gru``: 2-layer GRU width 96 over each FULL per-episode state
  sequence (padded), trained in minibatches of ``--ep-batch`` episodes per
  gradient step -- the pre-amendment version took ONE full-batch Adam step
  per epoch and early-stopped after 31-88 steps (reviewer-verified
  undertrained); loss/score only on masked steps.
- ``--mode mlp``: seeded 2x128 ReLU MLP on the caller-built design matrix
  ``X`` (the certificate's SACRED input format -- the flattened 4x14
  stride-2 XR1 window, or the 16-D pi0.5 single frame), minibatches of
  ``--row-batch`` masked rows per gradient step.

Both: Adam 1e-3, z-stats from fit rows, ``--max-epochs`` cap (default
300), early stop patience ``--patience`` (default 20) on held-out-train
R2 -- the hold-out is one train SEED (rng-chosen, seeded; its 3 conditions
give the per-episode-constant target nonzero validation variance),
``torch.manual_seed(SEED + fold)``, cudnn deterministic, wall budget
``--budget-s`` per cell.

Exchange npz (written by certificates.py):
  gru mode: states (N, D) float32 per-step sequences; ep_start (E+1,)
  int64 episode boundaries (contiguous, study row order).
  mlp mode: X (N, D) float32 design matrix.
  both: y (N,) float64 (mass_log_c); mask (N,) bool; seed (N,) int64;
  folds_json () str -- JSON [[train_seeds, test_seeds], ...] from the SAME
  masked-row GroupKFold the ridge path used.

Output npz: y_true/y_pred (pooled masked held-out, fold order), r2_folds,
epochs_per_fold, budget_hit, wall_s. Pooled R2 / rank_acc are computed by
the caller from y_true/y_pred so metric implementations stay
single-sourced in certificates.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

SEED = 0
GRU_HIDDEN = 96
GRU_LAYERS = 2
MLP_HIDDEN = 128
LR = 1e-3


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot


def _pick_val_seed(tr_seeds: list, np_rng) -> int:
    return int(sorted(tr_seeds)[int(np_rng.integers(len(tr_seeds)))])


def run_gru(data: dict, device: str, max_epochs: int, budget_s: float,
            patience: int, ep_batch: int) -> dict:
    import torch
    import torch.nn as nn

    states = np.asarray(data["states"], dtype=np.float32)
    y_all = np.asarray(data["y"], dtype=np.float64)
    mask_all = np.asarray(data["mask"], dtype=bool)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    ep_start = np.asarray(data["ep_start"], dtype=np.int64)
    folds = json.loads(str(data["folds_json"]))

    episodes = [(int(seeds[a]), slice(int(a), int(b)))
                for a, b in zip(ep_start[:-1], ep_start[1:])]
    Tmax = int(max(sl.stop - sl.start for _, sl in episodes))
    D = states.shape[1]

    def ep_tensors(sl):
        T = sl.stop - sl.start
        x = torch.zeros(Tmax, D)
        x[:T] = torch.from_numpy(states[sl])
        ym = torch.zeros(Tmax)
        ym[:T] = torch.from_numpy(y_all[sl].astype(np.float32))
        lm = torch.zeros(Tmax, dtype=torch.bool)
        lm[:T] = torch.from_numpy(mask_all[sl])
        vr = torch.zeros(Tmax, dtype=torch.bool)
        vr[:T] = True
        return x, ym, lm, vr

    cache = [ep_tensors(sl) for _, sl in episodes]

    t0 = time.time()
    torch.manual_seed(SEED)
    np_rng = np.random.default_rng(SEED)
    pooled_true, pooled_pred = [], []
    fold_r2s, fold_epochs = [], []
    budget_hit = False

    for fold, (tr_seeds, te_seeds) in enumerate(folds):
        tr_set, te_set = set(tr_seeds), set(te_seeds)
        val_seed = _pick_val_seed(tr_seeds, np_rng)
        has_mask = [bool(mask_all[sl].any()) for _, sl in episodes]
        fit_i = [i for i, (s, _) in enumerate(episodes)
                 if s in tr_set and s != val_seed and has_mask[i]]
        val_i = [i for i, (s, _) in enumerate(episodes) if s == val_seed and has_mask[i]]
        te_i = [i for i, (s, _) in enumerate(episodes) if s in te_set and has_mask[i]]

        def batch(ii):
            xs = torch.stack([cache[i][0] for i in ii]).to(device)
            ys = torch.stack([cache[i][1] for i in ii]).to(device)
            lm = torch.stack([cache[i][2] for i in ii]).to(device)
            vr = torch.stack([cache[i][3] for i in ii]).to(device)
            return xs, ys, lm, vr

        Xf, Yf, Mf, Vf = batch(fit_i)
        Xv, Yv, Mv, _ = batch(val_i)
        Xt, Yt, Mt, _ = batch(te_i)

        s_mu = Xf[Vf].mean(0)
        s_sd = Xf[Vf].std(0).clamp_min(1e-6)
        y_mu, y_sd = Yf[Mf].mean(), Yf[Mf].std().clamp_min(1e-6)

        torch.manual_seed(SEED + fold)
        gru = nn.GRU(input_size=D, hidden_size=GRU_HIDDEN, num_layers=GRU_LAYERS,
                     batch_first=True).to(device)
        head = nn.Linear(GRU_HIDDEN, 1).to(device)
        opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=LR)

        def forward(X):
            h, _ = gru((X - s_mu) / s_sd)
            return head(h).squeeze(-1)

        best_val, best_state, since_best = -np.inf, None, 0
        epoch = 0
        n_fit = len(fit_i)
        for epoch in range(max_epochs):
            if time.time() - t0 > budget_s:
                budget_hit = True
                break
            gru.train()
            order = torch.from_numpy(np_rng.permutation(n_fit))
            for lo in range(0, n_fit, ep_batch):  # minibatched (amendment B)
                sel = order[lo:lo + ep_batch]
                pred = forward(Xf[sel])
                m = Mf[sel]
                if not bool(m.any()):
                    continue
                loss = ((pred[m] - (Yf[sel][m] - y_mu) / y_sd) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            gru.eval()
            with torch.no_grad():
                vp = (forward(Xv) * y_sd + y_mu)[Mv].cpu().numpy()
            vt = Yv[Mv].cpu().numpy()
            val_r2 = _r2(vt, vp)
            crit = val_r2 if np.isfinite(val_r2) else -float(((vp - vt) ** 2).mean())
            if crit > best_val or best_state is None:
                best_val, since_best = crit, 0
                best_state = ([p.detach().clone() for p in gru.parameters()],
                              [p.detach().clone() for p in head.parameters()])
            else:
                since_best += 1
                if since_best >= patience:
                    break
        fold_epochs.append(epoch + 1)
        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(gru.parameters(), best_state[0]):
                    p.copy_(b)
                for p, b in zip(head.parameters(), best_state[1]):
                    p.copy_(b)
        gru.eval()
        with torch.no_grad():
            tp = (forward(Xt) * y_sd + y_mu)[Mt].cpu().numpy()
        tt = Yt[Mt].cpu().numpy()
        fold_r2s.append(_r2(tt, tp))
        pooled_true.append(tt)
        pooled_pred.append(tp)
        print(f"[gru-worker] fold {fold}: r2={fold_r2s[-1]:.4f} epochs={epoch + 1} "
              f"val_seed={val_seed} n_te={len(tt)}", flush=True)

    return _pack(pooled_true, pooled_pred, fold_r2s, fold_epochs, budget_hit, t0)


def run_mlp(data: dict, device: str, max_epochs: int, budget_s: float,
            patience: int, row_batch: int) -> dict:
    import torch
    import torch.nn as nn

    X_all = np.asarray(data["X"], dtype=np.float32)
    y_all = np.asarray(data["y"], dtype=np.float64)
    mask_all = np.asarray(data["mask"], dtype=bool)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    folds = json.loads(str(data["folds_json"]))

    t0 = time.time()
    torch.manual_seed(SEED)
    np_rng = np.random.default_rng(SEED)
    pooled_true, pooled_pred = [], []
    fold_r2s, fold_epochs = [], []
    budget_hit = False

    for fold, (tr_seeds, te_seeds) in enumerate(folds):
        tr_set, te_set = set(tr_seeds), set(te_seeds)
        val_seed = _pick_val_seed(tr_seeds, np_rng)
        rows_fit = np.flatnonzero(mask_all & np.isin(seeds, sorted(tr_set - {val_seed})))
        rows_val = np.flatnonzero(mask_all & (seeds == val_seed))
        rows_te = np.flatnonzero(mask_all & np.isin(seeds, sorted(te_set)))

        Xf = torch.from_numpy(X_all[rows_fit]).to(device)
        Yf = torch.from_numpy(y_all[rows_fit].astype(np.float32)).to(device)
        Xv = torch.from_numpy(X_all[rows_val]).to(device)
        Xt = torch.from_numpy(X_all[rows_te]).to(device)
        vt = y_all[rows_val]
        tt = y_all[rows_te]

        s_mu = Xf.mean(0)
        s_sd = Xf.std(0).clamp_min(1e-6)
        y_mu, y_sd = Yf.mean(), Yf.std().clamp_min(1e-6)

        torch.manual_seed(SEED + fold)
        mlp = nn.Sequential(
            nn.Linear(X_all.shape[1], MLP_HIDDEN), nn.ReLU(),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN), nn.ReLU(),
            nn.Linear(MLP_HIDDEN, 1),
        ).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=LR)

        def forward(X):
            return mlp((X - s_mu) / s_sd).squeeze(-1)

        best_val, best_state, since_best = -np.inf, None, 0
        epoch = 0
        n_fit = len(rows_fit)
        for epoch in range(max_epochs):
            if time.time() - t0 > budget_s:
                budget_hit = True
                break
            mlp.train()
            order = torch.from_numpy(np_rng.permutation(n_fit))
            for lo in range(0, n_fit, row_batch):  # minibatched (amendment B)
                sel = order[lo:lo + row_batch]
                loss = ((forward(Xf[sel]) - (Yf[sel] - y_mu) / y_sd) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            mlp.eval()
            with torch.no_grad():
                vp = (forward(Xv) * y_sd + y_mu).cpu().numpy()
            val_r2 = _r2(vt, vp)
            crit = val_r2 if np.isfinite(val_r2) else -float(((vp - vt) ** 2).mean())
            if crit > best_val or best_state is None:
                best_val, since_best = crit, 0
                best_state = [p.detach().clone() for p in mlp.parameters()]
            else:
                since_best += 1
                if since_best >= patience:
                    break
        fold_epochs.append(epoch + 1)
        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(mlp.parameters(), best_state):
                    p.copy_(b)
        mlp.eval()
        with torch.no_grad():
            tp = (forward(Xt) * y_sd + y_mu).cpu().numpy()
        fold_r2s.append(_r2(tt, tp))
        pooled_true.append(tt)
        pooled_pred.append(tp)
        print(f"[mlp-worker] fold {fold}: r2={fold_r2s[-1]:.4f} epochs={epoch + 1} "
              f"val_seed={val_seed} n_te={len(tt)}", flush=True)

    return _pack(pooled_true, pooled_pred, fold_r2s, fold_epochs, budget_hit, t0)


def _pack(pooled_true, pooled_pred, fold_r2s, fold_epochs, budget_hit, t0):
    return {
        "y_true": np.concatenate(pooled_true),
        "y_pred": np.concatenate(pooled_pred),
        "r2_folds": np.array(fold_r2s, dtype=np.float64),
        "epochs_per_fold": np.array(fold_epochs, dtype=np.int64),
        "budget_hit": np.array(budget_hit),
        "wall_s": np.array(round(time.time() - t0, 1)),
    }


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_npz")
    ap.add_argument("out_npz")
    ap.add_argument("--mode", choices=("gru", "mlp"), default="gru")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--ep-batch", type=int, default=16)
    ap.add_argument("--row-batch", type=int, default=256)
    args = ap.parse_args(argv)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with np.load(args.in_npz) as z:
        data = {k: z[k] for k in z.files}
    if args.mode == "gru":
        out = run_gru(data, args.device, args.max_epochs, args.budget_s,
                      args.patience, args.ep_batch)
    else:
        out = run_mlp(data, args.device, args.max_epochs, args.budget_s,
                      args.patience, args.row_batch)
    np.savez_compressed(args.out_npz, **out)
    print(f"[{args.mode}-worker] wrote {args.out_npz}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

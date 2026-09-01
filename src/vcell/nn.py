"""Embedding MLP on the additive model's residual -- a third model family.

Same target as ResidualBooster (the leading principal components of the residual),
different inductive bias: one shared trunk predicts all 96 components at once, so
5,920 training rows are amortised over the whole output instead of fitting 96
independent boosters.

Held-out entities are handled by construction.  Compound and strain embeddings are
zero-initialised and weight-decayed, so an entity that never appears in training
keeps the zero vector -- the "average entity" -- which is the same graceful
degradation the additive model has.  Everything else (plate, instrument, medium,
temperature, time, well position) is observed for every row, held out or not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FIELDS = [("Strains", 8, True), ("compound", 16, True), ("Yeast_cell_plate", 16, False),
          ("instrument", 4, False), ("data_source", 4, False), ("Medium", 2, False),
          ("Temperature", 2, False), ("pert_time", 4, False),
          ("well_row", 4, False), ("well_col", 4, False)]


def encode(meta: pd.DataFrame, levels: dict | None = None):
    """Integer-encode the design fields.  ``levels`` freezes the mapping so a
    second call on a different frame cannot renumber the categories."""
    m = meta.copy()
    m["well_row"] = m["protein_well"].str.extract(r"^([A-H])")[0]
    m["well_col"] = m["protein_well"].str.extract(r"(\d+)$")[0]
    fit = levels is None
    if fit:
        levels = {name: pd.Index(sorted(set(m[name].astype(str))))
                  for name, _, _ in FIELDS}
    cols = [levels[name].get_indexer(m[name].astype(str)) for name, _, _ in FIELDS]
    sizes = [len(levels[name]) for name, _, _ in FIELDS]
    return np.stack(cols, 1).astype(np.int64), sizes, levels


class ResidualMLP(nn.Module):
    def __init__(self, sizes, n_out, hidden=256, dropout=0.1):
        super().__init__()
        self.emb = nn.ModuleList()
        for (name, dim, zero), k in zip(FIELDS, sizes):
            e = nn.Embedding(k, dim)
            nn.init.zeros_(e.weight) if zero else nn.init.normal_(e.weight, 0, 0.05)
            self.emb.append(e)
        d = sum(dim for _, dim, _ in FIELDS)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(torch.cat([e(x[:, i]) for i, e in enumerate(self.emb)], 1))


class ResidualNN:
    """Fit/predict wrapper mirroring ResidualBooster's interface."""

    def __init__(self, n_comp: int = 96, hidden: int = 256, epochs: int = 400,
                 lr: float = 3e-3, weight_decay: float = 1e-4, dropout: float = 0.1,
                 batch: int = 256, n_seeds: int = 3, scale: float = 1.0,
                 threads: int = 8, verbose: bool = False):
        self.n_comp, self.hidden, self.epochs = n_comp, hidden, epochs
        self.lr, self.weight_decay, self.dropout = lr, weight_decay, dropout
        self.batch, self.n_seeds, self.scale = batch, n_seeds, scale
        self.threads, self.verbose = threads, verbose

    def fit(self, meta: pd.DataFrame, Y_obs: np.ndarray, use: np.ndarray,
            base: np.ndarray):
        torch.set_num_threads(self.threads)
        R = np.where(np.isfinite(Y_obs) & use[:, None], Y_obs - base, np.nan)
        Rv = np.nan_to_num(R[use]).astype(np.float32)
        U, S, Vt = np.linalg.svd(Rv, full_matrices=False)
        k = min(self.n_comp, Vt.shape[0])
        self.V = Vt[:k]
        self.explained = float((S[:k] ** 2).sum() / (S ** 2).sum())
        Z = (Rv @ self.V.T).astype(np.float32)
        self.zsd = Z.std(0) + 1e-6

        codes, sizes, self.levels = encode(meta)
        self.codes, self.sizes = codes, sizes
        Xv = torch.from_numpy(codes[use])
        Zt = torch.from_numpy(Z / self.zsd)

        self.models = []
        for seed in range(self.n_seeds):
            torch.manual_seed(seed)
            net = ResidualMLP(sizes, k, self.hidden, self.dropout)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, total_steps=self.epochs *
                max(1, len(Xv) // self.batch), pct_start=0.2)
            net.train()
            for ep in range(self.epochs):
                perm = torch.randperm(len(Xv))
                for i in range(0, len(Xv) - self.batch + 1, self.batch):
                    idx = perm[i:i + self.batch]
                    opt.zero_grad()
                    loss = ((net(Xv[idx]) - Zt[idx]) ** 2).mean()
                    loss.backward()
                    opt.step()
                    sched.step()
                if self.verbose and (ep + 1) % 100 == 0:
                    print(f"    seed {seed} epoch {ep+1} loss {loss.item():.4f}")
            net.eval()
            self.models.append(net)
        return self

    def predict(self, meta: pd.DataFrame | None = None) -> np.ndarray:
        codes = self.codes if meta is None else encode(meta, self.levels)[0]
        X = torch.from_numpy(codes)
        with torch.no_grad():
            Z = np.mean([m(X).numpy() for m in self.models], 0) * self.zsd
        return self.scale * (Z.astype(np.float32) @ self.V)

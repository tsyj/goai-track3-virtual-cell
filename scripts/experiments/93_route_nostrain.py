# -*- coding: utf-8 -*-
"""迭代 22：按实体可见性路由 booster——未见菌株行用"无 strain 特征"的 booster 成员（便宜池已有 Bns_boost_nostrain），
可见菌株行用常规成员；再做 v2.2 扩张。便宜池六折配对。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, evaluate, summary_row
OUT = os.path.join(ROOT, "results"); POOL = os.path.join(OUT, "pool_cheap")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
BETA, TAU = 0.7, 0.75

def ctx_ctrl(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def expand_unseen(P, meta, unseen):
    C, has, ctrl = ctx_ctrl(P, meta); D = np.where(has[:, None], P - C, 0.0)
    g = 1.0 + BETA * np.minimum(np.abs(D) / TAU, 1.0) ** 2
    return np.where((has & ~ctrl & unseen)[:, None], C + D * g, P).astype(np.float32)

rows = []
for seed, st in FOLDS:
    fo = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    A = np.load(os.path.join(POOL, f"{seed}_{st}__A.npy")); B = np.load(os.path.join(POOL, f"{seed}_{st}__Bns_boost_nostrain.npy"))
    unseen = (fo.meta["Strains"] == st).to_numpy()
    r = {"seed": seed, "strain": st}
    for name, P in [("A", A), ("Bns_all", B), ("route", np.where(unseen[:, None], B, A)), ("route_half", np.where(unseen[:, None], 0.5 * A + 0.5 * B, A))]:
        r[name] = summary_row("x", evaluate(fo, P.astype(np.float32), INNER))["TOTAL"]
        r[name + "+x"] = summary_row("x", evaluate(fo, expand_unseen(P.astype(np.float32), fo.meta, unseen), INNER))["TOTAL"]
    rows.append(r); print(f"  seed{seed} {st}", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "route_nostrain_cheap.csv"), index=False)
print(f"\n{'变体':12s} {'vs A':>10s} {'sem':>8s} {'up':>4s} {'| 加扩张后 vs A+x':>18s} {'sem':>8s} {'up':>4s}")
for k in ["Bns_all", "route", "route_half"]:
    dd = d[k] - d["A"]; de = d[k + "+x"] - d["A+x"]
    print(f"{k:12s} {dd.mean():+10.5f} {dd.sem():8.5f} {int((dd>0).sum()):>2d}/6 | {de.mean():+16.5f} {de.sem():8.5f} {int((de>0).sum()):>2d}/6")

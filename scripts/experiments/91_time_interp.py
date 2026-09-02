# -*- coding: utf-8 -*-
"""迭代 19：时间插值行（test_time 是插值不是外推）——同条件两侧时间点的训练残差线性插值，按 α 叠加。
条件键 = (data_source, Strains, Medium, Temperature, compound)。残差 r = y − P（P 已含板/仪器项，故 r 是去批次后的条件特异残差）。
纯缓存：val（val_time 行）+ 内层折（in_time 行）。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
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

def time_interp_correction(fo, P, target_mask):
    meta = fo.meta; tr = fo.obs_mask; R = np.where(tr[:, None] & np.isfinite(fo.Y), fo.Y - P, np.nan)
    key = meta[["data_source", "Strains", "Medium", "Temperature", "compound"]].astype(str).agg("|".join, axis=1).to_numpy()
    t = meta["pert_time"].astype(float).to_numpy()
    C = np.zeros_like(P); n_ok = 0
    df = pd.DataFrame({"k": key, "t": t, "i": np.arange(len(P)), "tr": tr})
    groups = {k: g for k, g in df[df.tr].groupby("k")}
    for i in np.where(target_mask)[0]:
        g = groups.get(key[i])
        if g is None: continue
        lo = g[g.t < t[i]]; hi = g[g.t > t[i]]
        if len(lo) == 0 or len(hi) == 0: continue
        tl, th = lo.t.max(), hi.t.min()
        rl = np.nanmean(R[lo[lo.t == tl].i.to_numpy()], 0); rh = np.nanmean(R[hi[hi.t == th].i.to_numpy()], 0)
        w = (t[i] - tl) / (th - tl)
        C[i] = np.nan_to_num((1 - w) * rl + w * rh); n_ok += 1
    return C, n_ok

def run(fo, Ps, held, which, target_split):
    meta = fo.meta; P0 = np.mean(Ps, 0).astype(np.float32)
    tmask = (meta["split_final"].astype(str) == target_split).to_numpy()
    C, n_ok = time_interp_correction(fo, P0, tmask)
    unseen = (meta["Strains"] == held).to_numpy(); out = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out[a] = summary_row("x", evaluate(fo, expand_unseen((P0 + a * C).astype(np.float32), meta, unseen), which))
    return out, n_ok, int(tmask.sum())

fo = build_fold(); Ps = [np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in MEMBERS]
res, n_ok, n_t = run(fo, Ps, "BAI", VAL, "val_time")
print(f"val_time 行 {n_t}，两侧有邻居的 {n_ok}")
for a, r in res.items(): print(f"  α={a:.2f} TOTAL {r['TOTAL']:.4f}  FC[time] {r['FC[time]']:.4f}  M5 {r['M5_bt(10%)']:.4f}  M1 {r['M1_abs(20%)']:.4f}")
rows = []
for seed, st in FOLDS:
    fi = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    Pi = [np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS]
    ri, n_ok, n_t = run(fi, Pi, st, INNER, "in_time")
    rows.append({"seed": seed, "strain": st, "n_ok": n_ok, "n_t": n_t, **{f"a{a}": r["TOTAL"] for a, r in ri.items()}})
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "time_interp_inner.csv"), index=False)
print("内层 in_time 行/有邻居:", list(zip(d.n_t, d.n_ok)))
for a in [0.25, 0.5, 0.75, 1.0]:
    dd = d[f"a{a}"] - d["a0.0"]; print(f"  α={a}: delta={dd.mean():+.5f} sem={dd.sem():.5f} up={(dd>0).sum()}/6")

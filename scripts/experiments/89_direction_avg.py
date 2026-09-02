# -*- coding: utf-8 -*-
"""迭代 17：集成聚合改到评分几何里——各成员的 D=P−C 先按样本归一化（除以该样本 D 的 rms）再平均，
幅度用各成员 rms 的均值还原；对照孔与无参照行保持算术平均。纯缓存，六折 + val（含 v2.2 扩张）。"""
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

def direction_avg(Ps, meta, mode):
    Pm = np.mean(Ps, 0).astype(np.float32)
    Cm, has, ctrl = ctx_ctrl(Pm, meta)
    Ds = []; rms = []
    for p in Ps:
        Cp, _, _ = ctx_ctrl(p.astype(np.float32), meta); d = p - Cp
        r = np.sqrt(np.nanmean(d ** 2, axis=1, keepdims=True)) + 1e-6
        Ds.append(d / r); rms.append(r)
    Dbar = np.mean(Ds, 0); Rbar = np.mean(rms, 0)
    if mode == "dir_mean_rms":
        Dn = Dbar * Rbar
    elif mode == "dir_rms_of_mean":                 # 幅度取算术平均 D 的 rms（保留成员间抵消）
        Dm = Pm - Cm; Dn = Dbar * (np.sqrt(np.nanmean(Dm ** 2, axis=1, keepdims=True)) + 1e-6)
    apply = has & ~ctrl
    return np.where(apply[:, None], Cm + Dn, Pm).astype(np.float32)

rows = []
for seed, st in FOLDS:
    fo = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    Ps = [np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS]
    unseen = (fo.meta["Strains"] == st).to_numpy()
    r = {"seed": seed, "strain": st, "v22": summary_row("x", evaluate(fo, expand_unseen(np.mean(Ps, 0).astype(np.float32), fo.meta, unseen), INNER))["TOTAL"]}
    for mode in ["dir_mean_rms", "dir_rms_of_mean"]:
        r[mode] = summary_row("x", evaluate(fo, expand_unseen(direction_avg(Ps, fo.meta, mode), fo.meta, unseen), INNER))["TOTAL"]
    rows.append(r); print(f"  seed{seed} {st}", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "direction_avg_inner.csv"), index=False)
fo = build_fold(); Ps = [np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in MEMBERS]
unseen = (fo.meta["Strains"] == "BAI").to_numpy()
v = {"v22": summary_row("x", evaluate(fo, expand_unseen(np.mean(Ps, 0).astype(np.float32), fo.meta, unseen), VAL))["TOTAL"]}
for mode in ["dir_mean_rms", "dir_rms_of_mean"]:
    v[mode] = summary_row("x", evaluate(fo, expand_unseen(direction_avg(Ps, fo.meta, mode), fo.meta, unseen), VAL))["TOTAL"]
print(f"\n{'变体':16s} {'内层 vs v2.2':>14s} {'sem':>8s} {'up':>4s} {'val':>8s} {'val vs v2.2':>11s}")
for k in ["dir_mean_rms", "dir_rms_of_mean"]:
    dd = d[k] - d["v22"]; print(f"{k:16s} {dd.mean():+14.5f} {dd.sem():8.5f} {int((dd>0).sum()):>2d}/6 {v[k]:8.4f} {v[k]-v['v22']:+11.4f}")

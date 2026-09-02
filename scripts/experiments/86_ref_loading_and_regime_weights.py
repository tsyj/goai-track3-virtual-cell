# -*- coding: utf-8 -*-
"""迭代 13/14（纯缓存，12 成员真配置，六折配对 + val 镜像）：
 A) 参照方向载荷校正：把 D̂ 在 μ_ctx（按上下文）/ μ_drug（按化合物）方向上的投影系数拉向 1
    D' = D + γ·(1−c)·μ，c = <D,μ>/<μ,μ>（逐样本），只在有参照的行；γ∈{0.5,1}
 B) 按行所属"体制"（可见/未见菌株 × 可见/未见化合物）分别给 F 族权重 w
"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
CUR = ["A","B","C","D","E","E16_pert16"]; FAM = ["F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
MEMBERS = CUR + FAM

def ctx_ctrl(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def expand_unseen(P, meta, unseen, beta=0.7, tau=0.75):
    C, has, ctrl = ctx_ctrl(P, meta); D = np.where(has[:, None], P - C, 0.0)
    g = 1.0 + beta * np.minimum(np.abs(D) / tau, 1.0) ** 2
    return np.where((has & ~ctrl & unseen)[:, None], C + D * g, P).astype(np.float32)

def ref_loading(P, fo, kind, gamma):
    """kind: 'ctx' 或 'drug'；参照 μ 用训练行的真实 Δ 均值（与评分器一致：只用训练行）"""
    meta = fo.meta; tr = fo.obs_mask
    Dtrue = fo.Y - fo.C_true
    key = (meta["ctx_key"].astype(str) if kind == "ctx" else meta["compound"].astype(str)).to_numpy()
    mu = {}
    df = pd.DataFrame({"k": key, "i": np.arange(len(P))})
    for k, g in df[tr].groupby("k"):
        mu[k] = np.nanmean(Dtrue[g.i.to_numpy()], 0)
    C, has, ctrl = ctx_ctrl(P, meta); D = P - C
    out = P.copy()
    for k, g in df.groupby("k"):
        if k not in mu: continue
        m = np.nan_to_num(mu[k]); mm = float(m @ m)
        if mm < 1e-6: continue
        idx = g.i.to_numpy(); idx = idx[has[idx] & ~ctrl[idx]]
        if len(idx) == 0: continue
        c = (np.nan_to_num(D[idx]) @ m) / mm
        out[idx] = P[idx] + gamma * (1.0 - c)[:, None] * m[None, :]
    return out.astype(np.float32)

def regime_blend(Ps_cur, Ps_fam, meta, unseen_strain, unseen_cmpd, W):
    Pc = np.mean(Ps_cur, 0); Pf = np.mean(Ps_fam, 0)
    w = np.full(len(meta), 0.5, np.float32)
    w[~unseen_strain & ~unseen_cmpd] = W["ss"]; w[unseen_strain & ~unseen_cmpd] = W["us"]
    w[~unseen_strain & unseen_cmpd] = W["sc"]; w[unseen_strain & unseen_cmpd] = W["uc"]
    return ((1 - w)[:, None] * Pc + w[:, None] * Pf).astype(np.float32)

WGRID = {"eq": dict(ss=.5, us=.5, sc=.5, uc=.5), "us.7": dict(ss=.5, us=.7, sc=.5, uc=.7), "us.3": dict(ss=.5, us=.3, sc=.5, uc=.3),
         "sc.7": dict(ss=.5, us=.5, sc=.7, uc=.5), "sc.3": dict(ss=.5, us=.5, sc=.3, uc=.5), "ss.7": dict(ss=.7, us=.5, sc=.5, uc=.5), "ss.3": dict(ss=.3, us=.5, sc=.5, uc=.5)}

def run(fo, Ps_cur, Ps_fam, held, tag):
    meta = fo.meta; unseen = (meta["Strains"] == held).to_numpy()
    sp = meta["split_final"].astype(str).to_numpy()
    unseen_c = np.isin(sp, ["in_chem_only", "in_both", "val_chem_only", "val_both"])
    P = np.mean(Ps_cur + Ps_fam, 0).astype(np.float32)
    base = expand_unseen(P, meta, unseen)
    out = {"v22": summary_row("x", evaluate(fo, base, tag))["TOTAL"]}
    for kind in ["ctx", "drug"]:
        for gm in [0.5, 1.0]:
            Q = expand_unseen(ref_loading(P, fo, kind, gm), meta, unseen)
            out[f"ref_{kind}_g{gm}"] = summary_row("x", evaluate(fo, Q, tag))["TOTAL"]
    for name, W in WGRID.items():
        Q = expand_unseen(regime_blend(Ps_cur, Ps_fam, meta, unseen, unseen_c, W), meta, unseen)
        out[f"w_{name}"] = summary_row("x", evaluate(fo, Q, tag))["TOTAL"]
    return out

rows = []
for seed, st in FOLDS:
    fo = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    Pc = [np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in CUR]; Pf = [np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in FAM]
    r = run(fo, Pc, Pf, st, INNER); r.update({"seed": seed, "strain": st}); rows.append(r); print(f"  seed{seed} {st}", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "ref_loading_regime_inner.csv"), index=False)
fo = build_fold(); Pc = [np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in CUR]; Pf = [np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in FAM]
v = run(fo, Pc, Pf, "BAI", VAL); pd.Series(v).to_csv(os.path.join(OUT, "ref_loading_regime_val.csv"))
print(f"\n{'变体':14s} {'内层 vs v2.2':>14s} {'sem':>8s} {'up':>4s} {'val':>8s} {'val vs v2.2':>11s}")
for k in [c for c in d.columns if c not in ("seed", "strain", "v22")]:
    dd = d[k] - d["v22"]; print(f"{k:14s} {dd.mean():+14.5f} {dd.sem():8.5f} {int((dd>0).sum()):>2d}/6 {v[k]:8.4f} {v[k]-v['v22']:+11.4f}")

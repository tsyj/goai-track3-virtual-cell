# -*- coding: utf-8 -*-
"""迭代 11：已采纳的尾部扩张 h(D) 的精修变体（12 成员真配置缓存，六折配对 + val 镜像）。
  base      : 无扩张
  adopted   : β=0.3 τ=1.25，作用于集成均值
  permember : 同参数，逐成员先扩张再平均（非线性→不等价）
  tau_prot  : τ 按蛋白定（τ_j = 2.5×该蛋白训练行 Δ 的 sd，剪到 [0.5, 3]）
  asym_up   : 只放大上调（D>0）
  asym_down : 只放大下调（D<0）
  seen_only : 只对可见菌株的行扩张（未见菌株行不动）
  unseen_only: 只对未见菌株行扩张
"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
BETA, TAU = 0.3, 1.25

def ctx_ctrl(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def expand(P, meta, rows_mask=None, tau=TAU, beta=BETA, sign=None):
    C, has, ctrl = ctx_ctrl(P, meta); D = np.where(has[:, None], P - C, 0.0)
    t = tau if np.isscalar(tau) else tau[None, :]
    g = 1.0 + beta * np.minimum(np.abs(D) / t, 1.0) ** 2
    if sign == "up": g = np.where(D > 0, g, 1.0)
    if sign == "down": g = np.where(D < 0, g, 1.0)
    apply = has & ~ctrl
    if rows_mask is not None: apply = apply & rows_mask
    return np.where(apply[:, None], C + D * g, P).astype(np.float32)

def variants(fo, Ps, held):
    meta = fo.meta; P = np.mean(Ps, 0).astype(np.float32)
    # 按蛋白 τ：训练行 Δ 的 sd
    tr = fo.obs_mask; Dtr = (fo.Y - fo.C_true)[tr]
    sd = np.nanstd(Dtr, axis=0); tau_p = np.clip(2.5 * np.where(np.isfinite(sd), sd, 0.5), 0.5, 3.0).astype(np.float32)
    unseen = (meta["Strains"] == held).to_numpy() if held else np.zeros(len(meta), bool)
    return {
        "base": P,
        "adopted": expand(P, meta),
        "permember": np.mean([expand(p.astype(np.float32), meta) for p in Ps], 0).astype(np.float32),
        "tau_prot": expand(P, meta, tau=tau_p),
        "asym_up": expand(P, meta, sign="up"),
        "asym_down": expand(P, meta, sign="down"),
        "seen_only": expand(P, meta, rows_mask=~unseen),
        "unseen_only": expand(P, meta, rows_mask=unseen),
    }

rows = []
for seed, st in FOLDS:
    fo = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    Ps = [np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS]
    for name, Pn in variants(fo, Ps, st).items():
        r = summary_row(name, evaluate(fo, Pn, INNER)); r.update({"seed": seed, "strain": st, "variant": name}); rows.append(r)
    print(f"  seed{seed} {st} done", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "expand_variants_inner.csv"), index=False)
piv = d.pivot_table(index=["seed","strain"], columns="variant", values="TOTAL")
print("\n=== 内层六折（vs adopted）===")
for v in ["base","permember","tau_prot","asym_up","asym_down","seen_only","unseen_only"]:
    dd = piv[v] - piv["adopted"]; print(f"  {v:12s} delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
# val 镜像
fo = build_fold()
Ps = [np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in MEMBERS]
print("\n=== 官方 val 镜像 ===")
res = {}
for name, Pn in variants(fo, Ps, "BAI").items():
    r = summary_row(name, evaluate(fo, Pn, VAL)); res[name] = r["TOTAL"]
for name, t in res.items():
    print(f"  {name:12s} TOTAL {t:.4f}  (vs adopted {t-res['adopted']:+.4f})")

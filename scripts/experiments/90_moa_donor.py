# -*- coding: utf-8 -*-
"""迭代 18：未见化合物的"机制类供体"迁移（专家 MOA 类表，不是化学相似度）。
对留出化合物 c 的处理行：P' = P + κ·(μ_drug[donor] − μ̄)，μ_drug = 训练行该化合物 Δ 均值，μ̄ = 训练化合物 μ_drug 的均值
（先验：我们对未见化合物的预测 ≈ 通用响应；供体给出其机制特异偏离）。val 镜像（留出化合物 6 个，4 个有供体）+ 内层折。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
from vcell.metrics import _pcc_rows
OUT = os.path.join(ROOT, "results")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
BETA, TAU = 0.7, 0.75
CLASSES = {
    "dna_damage": ["Cisplatin", "Hydroxyurea", "MMS", "(S)-(+)-Camptothecin"],
    "azole": ["Clotrimazole", "Fluconazole"],
    "polyene": ["Nystatin dihydrate", "Amphotericin B"],
    "translation": ["CHX", "Anisomycin", "G418", "Hygromycin B", "Neomycin B"],
    "serm": ["4-Hydroxytamoxifen", "Clomiphene citrate", "Tamoxifen", "Raloxifene hydrochloride"],
    "ionophore": ["Nigericin", "Valinomycin", "FCCP"],
    "oxidative": ["Artemisinin", "Parthenolide", "Emodin", "H2O2", "Plumbagin"],
}
CLS = {c: k for k, v in CLASSES.items() for c in v}

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

def donor_correction(fo):
    meta = fo.meta; tr = fo.obs_mask; Dt = fo.Y - fo.C_true
    comp = meta["compound"].astype(str).to_numpy(); ctrl = meta["is_control"].to_numpy()
    mu = {}
    for c in np.unique(comp[tr & ~ctrl]):
        m = tr & (comp == c); mu[c] = np.nanmean(Dt[m], 0)
    mubar = np.nanmean(np.stack(list(mu.values())), 0)
    seen = set(mu)
    C = np.zeros((len(meta), Dt.shape[1]), np.float32); applied = {}
    for c in np.unique(comp[~tr & ~ctrl]):
        if c in seen or c not in CLS: continue
        donors = [d for d in CLASSES[CLS[c]] if d in seen and d != c]
        if not donors: continue
        dev = np.nanmean(np.stack([mu[d] - mubar for d in donors]), 0)
        rows = (~tr) & (comp == c)
        C[rows] = np.nan_to_num(dev); applied[c] = (donors, int(rows.sum()))
    return C, applied

def run(tag, fo, Ps, held, which):
    meta = fo.meta; P0 = np.mean(Ps, 0).astype(np.float32)
    C, applied = donor_correction(fo)
    unseen = (meta["Strains"] == held).to_numpy()
    out = {}
    for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
        P = expand_unseen((P0 + k * C).astype(np.float32), meta, unseen)
        r = summary_row("x", evaluate(fo, P, which)); out[k] = r
    return out, applied, P0, C

# ---- val
fo = build_fold(); Ps = [np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in MEMBERS]
res, applied, P0, C = run("val", fo, Ps, "BAI", VAL)
print("val 供体映射:", {c: (d, n) for c, (d, n) in applied.items()})
print(f"{'k':>5s} {'TOTAL':>8s} {'M2':>8s} {'M3':>8s} {'FC[chem]':>9s} {'FC[both]':>9s}")
for k, r in res.items():
    print(f"{k:5.2f} {r['TOTAL']:8.4f} {r['M2_rawFC(25%)']:8.4f} {r['M3_ctx(20%)']:8.4f} {r['FC[chem_only]']:9.4f} {r['FC[both]']:9.4f}")
# 逐化合物 Δ-PCC（val 未见化合物）
meta = fo.meta; Dt = fo.Y - fo.C_true; valun = meta.split_final.isin(["val_chem_only", "val_both"]).to_numpy()
for c in applied:
    rows = valun & (meta.compound == c).to_numpy()
    for k in [0.0, 0.5, 1.0]:
        Dp = (P0 + k * C) - fo.C_true
        print(f"  {c[:22]:22s} k={k}: Δ-PCC {np.nanmean(_pcc_rows(Dp[rows], Dt[rows])):.3f}")
# ---- inner
rows_out = []
for seed, st in FOLDS:
    fi = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    Pi = [np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS]
    ri, ap, _, _ = run("inner", fi, Pi, st, INNER)
    rows_out.append({"seed": seed, "strain": st, "n_applied": sum(n for _, n in ap.values()), **{f"k{k}": r["TOTAL"] for k, r in ri.items()}})
d = pd.DataFrame(rows_out); d.to_csv(os.path.join(OUT, "moa_donor_inner.csv"), index=False)
print("\n内层折（留出化合物里有供体的行数）:", d.n_applied.tolist())
for k in [0.25, 0.5, 0.75, 1.0]:
    dd = d[f"k{k}"] - d["k0.0"]; print(f"  k={k}: delta={dd.mean():+.5f} sem={dd.sem():.5f} up={(dd>0).sum()}/6")

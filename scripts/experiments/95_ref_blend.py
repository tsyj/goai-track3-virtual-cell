# -*- coding: utf-8 -*-
"""迭代 23/24：向冻结参照混合 + 蛋白轴 γ<1 校准。
关键性质：D' = λD + (1−λ)μ 使 (D'−μ) = λ(D−μ)，PCC 尺度不变 ⇒ M3/M4 恒不变，只动 M2/M1/M5/M6。
μ 全部只用 split_final=='train' 行的 Δ_true 计算（与评分器同键，但来源合法）。
内层六折 + val 镜像，纯缓存。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
MEM = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
BETA, TAU = 0.7, 0.75
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
LAM = [1.0, 0.7, 0.85, 1.15, 1.3, 1.5, 1.8]
GAM = [1.0, 0.98, 0.95, 0.92, 0.88]

def ctx_ctrl(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def mu_of(fo, key):
    meta = fo.meta; tr = fo.obs_mask; Dt = fo.Y - fo.C_true
    k = meta[key].astype(str).to_numpy(); tab = {}
    df = pd.DataFrame({"k": k, "i": np.arange(len(meta))})
    for kk, g in df[tr].groupby("k"):
        v = np.nanmean(Dt[g.i.to_numpy()], 0)
        if np.isfinite(v).any(): tab[kk] = np.nan_to_num(v)
    return k, tab

def mu_global(fo):
    Dt = fo.Y - fo.C_true; tr = fo.obs_mask
    m = fo.meta; treated = tr & ~m["is_control"].to_numpy() & ~m["is_qc"].to_numpy()
    return np.nan_to_num(np.nanmean(Dt[treated], 0))


def transform(P, fo, held, lam_chem, lam_strain, gamma, lam_both=1.0):
    meta = fo.meta; C, has, ctrl = ctx_ctrl(P, meta); D = np.where(has[:, None], P - C, 0.0)
    sp = meta["split_final"].astype(str).to_numpy()
    chem = np.isin(sp, ["in_chem_only", "val_chem_only"]); strn = np.isin(sp, ["in_strain_only", "val_strain_only"])
    kc, mc = mu_of(fo, "ctx_key"); kd, md = mu_of(fo, "compound")
    Dn = D.copy()
    if lam_chem < 1.0:
        rows = np.where(has & ~ctrl & chem)[0]
        ok = [i for i in rows if kc[i] in mc]
        if ok: Dn[ok] = lam_chem*D[ok] + (1-lam_chem)*np.stack([mc[kc[i]] for i in ok])
    if lam_strain < 1.0:
        rows = np.where(has & ~ctrl & strn)[0]
        ok = [i for i in rows if kd[i] in md]
        if ok: Dn[ok] = lam_strain*D[ok] + (1-lam_strain)*np.stack([md[kd[i]] for i in ok])
    if lam_both != 1.0:
        both = np.isin(sp, ["in_both", "val_both"])
        rows = np.where(has & ~ctrl & both)[0]
        if len(rows): mg = mu_global(fo); Dn[rows] = lam_both*D[rows] + (1-lam_both)*mg[None, :]
    unseen = (meta["Strains"] == held).to_numpy()
    g = 1.0 + BETA*np.minimum(np.abs(Dn)/TAU, 1.0)**2
    Dn = np.where(unseen[:, None], Dn*g, Dn)
    Q = np.where((has & ~ctrl)[:, None], C + Dn, P).astype(np.float32)
    if gamma != 1.0:
        mj = Q[fo.obs_mask].mean(0); Q = (mj[None, :] + gamma*(Q - mj[None, :])).astype(np.float32)
    return Q

def run(fo, P, held, which):
    def sc(**kw):
        r = summary_row("x", evaluate(fo, transform(P, fo, held, kw.get("lc",1.0), kw.get("ls",1.0), kw.get("g",1.0), kw.get("lb",1.0)), which))
        return r
    base = sc(); out = {"v22": base["TOTAL"]}
    for m in ["M1_abs(20%)","M2_rawFC(25%)","M3_ctx(20%)","M4_drug(20%)"]: out["v22|"+m] = base[m]
    for l in LAM[1:]:
        r = sc(lc=l); out[f"chem{l}"] = r["TOTAL"]; out[f"chem{l}|M3"] = r["M3_ctx(20%)"]; out[f"chem{l}|M2"] = r["M2_rawFC(25%)"]
        r = sc(ls=l); out[f"strain{l}"] = r["TOTAL"]; out[f"strain{l}|M4"] = r["M4_drug(20%)"]; out[f"strain{l}|M2"] = r["M2_rawFC(25%)"]
        out[f"both{l}"] = sc(lb=l)["TOTAL"]
    for g in GAM[1:]: out[f"gam{g}"] = sc(g=g)["TOTAL"]
    return out

fo = build_fold()
Pv = np.mean([np.load(f"{OUT}/val_parts/{m}_add.npy") + np.load(f"{OUT}/val_parts/{m}_boost.npy") for m in MEM], 0).astype(np.float32)
v = run(fo, Pv, "BAI", VAL)
rows = []
for seed, st in FOLDS:
    fi = pickle.load(open(f"{OUT}/folds/{seed}_{st}.pkl", "rb"))
    Pi = np.mean([np.load(f"{OUT}/pool_real/{seed}_{st}__{m}.npy") for m in MEM], 0).astype(np.float32)
    r = run(fi, Pi, st, INNER); r.update({"seed": seed, "strain": st}); rows.append(r); print(f"  seed{seed} {st}", flush=True)
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/ref_blend_inner.csv", index=False); pd.Series(v).to_csv(f"{OUT}/ref_blend_val.csv")
print(f"\n{'变体':12s} {'内层 vs v2.2':>13s} {'sem':>8s} {'up':>4s} {'ratio':>6s} {'val':>8s} {'val vs v2.2':>11s}")
for k in [c for c in d.columns if c not in ("seed","strain","v22") and "|" not in c]:
    dd = d[k] - d["v22"]
    print(f"{k:12s} {dd.mean():+13.5f} {dd.sem():8.5f} {int((dd>0).sum()):>2d}/6 {dd.mean()/max(dd.sem(),1e-9):6.1f} {v[k]:8.4f} {v[k]-v['v22']:+11.4f}")

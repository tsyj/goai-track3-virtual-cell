# -*- coding: utf-8 -*-
"""迭代 15：用公开【实测蛋白质组】（PNAS 2024，942 株 × 888 蛋白，SC 培养基，Scanning-SWATH）
直接读出未见菌株的每蛋白基线偏移 b，而不是从基因型预测。

与 63–65（PAV/CNV/移码 → b）机制上本质不同：这是 b 的同一物理量在另一平台上的测量值。
步骤（与 65 同构，全部只用训练行 + 缓存预测，不碰测试真值）：
  1. PNAS 表：取本题菌株列（BAH/BAI/CEK/CGD/CRD；DHY210 不在），log2，按蛋白在"该折可见且在 PNAS 中"的菌株上中心化
     → a_s,j = log2 x_s,j − mean_{visible} log2 x_·,j
  2. 竞赛侧：可见菌株 s 的 b_s,j = mean(该菌株训练行) − mean(其它可见菌株训练行)（同 65）
  3. 留一菌株：用其它可见菌株拟合单一标量 κ（b ≈ κ·a），报告留出株上的 R²/相关
  4. 修正 C = κ·a_held（只在有 PNAS 覆盖的蛋白上），加到留出菌株的全部行，再做 v2.2 的尾部扩张，逐折配对评分 k∈SCALES
  5. 官方 val 镜像（留出 BAI）终验

    python scripts/87_external_proteome_b.py
"""
import glob
import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, make_inner_splits, evaluate, summary_row  # noqa: E402

RES = os.path.join(ROOT, "results")
EXT = os.path.join(ROOT, "data", "external", "pnas2024")
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
MEMBERS = ["A", "B", "C", "D", "E", "E16_pert16", "F_strain_early", "FB_early_plate1",
           "FC_early_plate4", "FD_early_pert2", "FE_early_pert8", "FE16_early_pert16"]
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
BETA, TAU = 0.7, 0.75

_s = importlib.util.spec_from_file_location("g63", os.path.join(ROOT, "scripts", "63_genome_pav_diagnostic.py"))
g63 = importlib.util.module_from_spec(_s); _s.loader.exec_module(g63)


def load_pnas():
    cands = glob.glob(os.path.join(EXT, "**", "*DetectionThreshold30_genes_ORF*.tsv"), recursive=True) or glob.glob(os.path.join(EXT, "**", "*SCmedia*DIA-NN*genes_ORF*.tsv"), recursive=True) or \
            glob.glob(os.path.join(EXT, "**", "*ProteomicsData*ORF*.tsv"), recursive=True)
    assert cands, f"未找到 PNAS 蛋白质组 tsv（{EXT}）"
    df = pd.read_csv(cands[0], sep="\t")
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str).str.upper()
    print(f"PNAS 表 {os.path.basename(cands[0])}: {df.shape}，菌株列含 " +
          ", ".join(s for s in ["BAH", "BAI", "CEK", "CGD", "CRD", "DHY210"] if s in df.columns), flush=True)
    X = df[[c for c in ["BAH", "BAI", "CEK", "CGD", "CRD"] if c in df.columns]].astype(float)
    X = X.where(X > 0)
    return np.log2(X)          # ORF × strain


def orf_index(proteins):
    """竞赛蛋白列 → ORF 系统名（用 63 的 SGD 映射）"""
    sym = g63.sgd_symbol_to_orf()
    return [sym.get(p.upper()) for p in proteins]


def b_visible(fold, s):
    meta, Y, obs = fold.meta, fold.Y_obs, fold.obs_mask
    rs = obs & (meta["Strains"] == s).to_numpy()
    ot = obs & (meta["Strains"] != s).to_numpy()
    return np.nanmean(np.where(rs[:, None], Y, np.nan), 0) - np.nanmean(np.where(ot[:, None], Y, np.nan), 0)


def a_external(L, orfs, s, ref_strains):
    """a_s,j = log2 x_s − mean(ref) 在竞赛蛋白顺序上；无覆盖处 NaN"""
    out = np.full(len(orfs), np.nan)
    ref = L[ref_strains].mean(axis=1)
    for j, o in enumerate(orfs):
        if o is not None and o in L.index and s in L.columns and np.isfinite(L.at[o, s]) and np.isfinite(ref.get(o, np.nan)):
            out[j] = L.at[o, s] - ref[o]
    return out


def fit_kappa(fold, L, orfs, held):
    """留一可见菌株拟合 κ；返回 κ 与每株的留出相关/R²"""
    vis = [s for s in fold.meta["Strains"].unique() if s != held and s in L.columns]
    rows = []
    xs, ys = [], []
    for s in vis:
        others = [t for t in vis if t != s]
        if not others:
            continue
        b = b_visible(fold, s); a = a_external(L, orfs, s, others)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 30:
            continue
        # 拟合 κ 时留出 s
        kap_s = None
        xo, yo = [], []
        for t in others:
            bt = b_visible(fold, t); at = a_external(L, orfs, t, [u for u in vis if u != t])
            m = np.isfinite(at) & np.isfinite(bt); xo.append(at[m]); yo.append(bt[m])
        if xo:
            xo = np.concatenate(xo); yo = np.concatenate(yo)
            kap_s = float((xo * yo).sum() / max((xo * xo).sum(), 1e-9))
            pred = kap_s * a[ok]
            r2 = 1 - ((b[ok] - pred) ** 2).sum() / ((b[ok] - b[ok].mean()) ** 2).sum()
            rows.append({"strain": s, "n_cov": int(ok.sum()), "corr": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                         "kappa_loso": kap_s, "r2_loso": float(r2)})
        xs.append(a[ok]); ys.append(b[ok])
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    kappa = float((xs * ys).sum() / max((xs * xs).sum(), 1e-9))
    return kappa, pd.DataFrame(rows)


def expand_unseen(P, meta, unseen):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    D = np.where(has[:, None], P - C, 0.0)
    gm = 1.0 + BETA * np.minimum(np.abs(D) / TAU, 1.0) ** 2
    return np.where((has & ~ctrl & unseen)[:, None], C + D * gm, P).astype(np.float32)


def run(tag, folds, which):
    L = load_pnas()
    out, diag = [], []
    for seed, held in folds:
        fo = build_fold() if tag == "val" else build_fold(splits=make_inner_splits(build_fold().meta, hold_strain=held, seed=seed))
        proteins = [str(p) for p in fo.proteins]; orfs = orf_index(proteins)
        kappa, d = fit_kappa(fo, L, orfs, held); d["seed"] = seed; d["held"] = held; diag.append(d)
        vis = [s for s in fo.meta["Strains"].unique() if s != held and s in L.columns]
        a_h = a_external(L, orfs, held, vis)
        cov = np.isfinite(a_h)
        delta = np.where(cov, kappa * a_h, 0.0).astype(np.float32)
        rows = (fo.meta["Strains"] == held).to_numpy()
        C = np.zeros((len(fo.meta), len(proteins)), np.float32); C[rows] = delta
        if tag == "val":
            P0 = np.mean([np.load(os.path.join(RES, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(RES, "val_parts", f"{m}_boost.npy")) for m in MEMBERS], 0).astype(np.float32)
        else:
            P0 = np.mean([np.load(os.path.join(RES, "pool_real", f"{seed}_{held}__{m}.npy")) for m in MEMBERS], 0).astype(np.float32)
        print(f"[{tag}] seed{seed} {held}: κ={kappa:.3f} 覆盖蛋白 {int(cov.sum())}/{len(proteins)} "
              f"|δ| rms {np.sqrt((delta[cov]**2).mean()):.3f}  " + "  ".join(f"{r.strain}:corr={r['corr']:.2f},R²={r.r2_loso:.3f}" for _, r in d.iterrows()), flush=True)
        unseen = rows & ~fo.obs_mask
        for k in SCALES:
            P = expand_unseen((P0 + k * C).astype(np.float32), fo.meta, unseen)
            r = summary_row(f"k={k}", evaluate(fo, P, which)); r.update({"seed": seed, "strain": held, "k": k}); out.append(r)
    return pd.DataFrame(out), pd.concat(diag, ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    print("=== 六个无孤儿内层折 ===", flush=True)
    inner, diag = run("inner", FOLDS, INNER)
    inner.to_csv(os.path.join(RES, "extprot_corr_inner.csv"), index=False); diag.to_csv(os.path.join(RES, "extprot_diag.csv"), index=False)
    piv = inner.pivot_table(index=["seed", "strain"], columns="k", values="TOTAL"); base = piv[0.0]
    for k in SCALES[1:]:
        dd = piv[k] - base; print(f"  k={k:<5} delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
    for mod in ["M1_abs(20%)", "M2_rawFC(25%)", "M4_drug(20%)", "FC[strain_only]", "FC[both]"]:
        q = inner.pivot_table(index=["seed", "strain"], columns="k", values=mod)
        print(f"  {mod:18s} " + "  ".join(f"k={c}:{(q[c]-q[0.0]).mean():+.5f}" for c in q.columns if c != 0.0))
    print("\n=== 官方 val 镜像（留出 BAI）===", flush=True)
    val, dv = run("val", [(0, "BAI")], VAL)
    val.to_csv(os.path.join(RES, "extprot_corr_val.csv"), index=False)
    print(val[["k", "TOTAL", "FC[strain_only]", "FC[both]", "M1_abs(20%)", "M2_rawFC(25%)", "M4_drug(20%)"]].round(4).to_string(index=False))

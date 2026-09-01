"""Genome-informed baseline correction for the unseen strain -- scored on the folds.

Established in scripts/63 and 64, on training rows only:

    gene absent (PAV=0)        b = -0.337   (n=235)
    homozygous frameshift      b = -0.412   (n=20)
    copy number < 0.6          b = -0.224   (n=365)
    everything else            b = -0.033   (n=12,517)

b is the per-(strain, protein) baseline shift that docs/FINDINGS §2 identified as
the dominant error on the unseen-strain splits and declared unestimable.  It is
unestimable *from labels*.  The 1,011 Yeast Genomes release gives, for every one
of our strains including the officially held-out CRD, which genes the isolate is
missing, which are frameshifted and which are present at reduced copy number --
no labels involved.

Every strain-indexed term of the additive model evaluates to zero on a held-out
strain, so that strain currently receives no strain-specific signal at all.  This
script adds one: b_hat[p] built from the genome flags, applied ONLY to rows whose
strain has no training label in that fold, with coefficients fitted on the other
(visible) strains of the same fold -- i.e. genuinely leave-one-strain-out.

Scored by replaying the cached member predictions in results/pool_real (six
orphan-free inner folds) and results/pool_val (official val mirror, held-out
strain BAI), so no model is refitted.

    python scripts/65_genome_baseline_correction.py

Jiao Xinyuan 2026-09-02
"""
import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import (INNER, VAL, build_fold, evaluate,      # noqa: E402
                           make_inner_splits, summary_row)

_s = importlib.util.spec_from_file_location("g64", os.path.join(ROOT, "scripts", "64_genome_explains_b.py"))
g64 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(g64)

RES = os.path.join(ROOT, "results")
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
SHIPPED = ["A", "B", "C", "D", "E", "E16_pert16", "F_strain_early", "FB_early_plate1",
           "FC_early_plate4", "FD_early_pert2", "FE_early_pert8", "FE16_early_pert16"]
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]


def genome_tables():
    pav = g64.strain_by_orf(os.path.join(g64.GEN, "genesMatrix_PresenceAbsence.tab.gz"))
    cnv = g64.strain_by_orf(os.path.join(g64.GEN, "genesMatrix_CopyNumber.tab.gz"))
    try:
        fs = g64.orf_by_strain(os.path.join(g64.GEN, "genesMatrix_Frameshift.tab.gz"), "Gene", None)
    except Exception:
        fs = None
    return pav, cnv, fs


def flags_for(strain, orfs, pav, cnv, fs):
    """(3, n_proteins) indicator matrix: [absent, frameshift, low copy number]."""
    n = len(orfs)
    A = np.zeros((3, n), np.float32)
    if strain not in pav.index:
        return A                       # DHY210: lab strain, S288c proxy => reference, no flags
    for j, o in enumerate(orfs):
        if o is None:
            continue
        if o in pav.columns and np.isfinite(pav.loc[strain, o]) and pav.loc[strain, o] < 0.5:
            A[0, j] = 1.0
        if fs is not None and strain in fs.index and o in fs.columns:
            v = fs.loc[strain, o]
            if np.isfinite(v) and v > 0.5:
                A[1, j] = 1.0
        if o in cnv.columns and np.isfinite(cnv.loc[strain, o]) and cnv.loc[strain, o] < 0.6:
            A[2, j] = 1.0
    return A


def fit_coeffs(fold, proteins, orfs, pav, cnv, fs, held):
    """Per-flag mean of b, estimated on the fold's VISIBLE strains only."""
    meta, Y, obs = fold.meta, fold.Y_obs, fold.obs_mask
    strains = [s for s in meta["Strains"].unique() if s != held]
    num = np.zeros(3); den = np.zeros(3); base_n = 0; base_s = 0.0
    for s in strains:
        rs = obs & (meta["Strains"] == s).to_numpy()
        ot = obs & (meta["Strains"] != s).to_numpy()
        if rs.sum() == 0 or ot.sum() == 0:
            continue
        b = np.nanmean(np.where(rs[:, None], Y, np.nan), 0) - \
            np.nanmean(np.where(ot[:, None], Y, np.nan), 0)
        A = flags_for(s, orfs, pav, cnv, fs)
        ok = np.isfinite(b)
        for k in range(3):
            m = ok & (A[k] > 0.5)
            num[k] += b[m].sum(); den[k] += m.sum()
        rest = ok & (A.sum(0) < 0.5)
        base_s += b[rest].sum(); base_n += rest.sum()
    base = base_s / max(base_n, 1)
    coef = np.where(den > 0, num / np.maximum(den, 1) - base, 0.0)
    return coef, base, den


def correction(fold, proteins, orfs, pav, cnv, fs, coef, held):
    """(n, p) additive correction applied only to rows of the label-free strain."""
    meta = fold.meta
    C = np.zeros((len(meta), len(proteins)), np.float32)
    A = flags_for(held, orfs, pav, cnv, fs)
    delta = (coef[:, None] * A).sum(0)
    rows = (meta["Strains"] == held).to_numpy() & ~fold.obs_mask
    C[rows] = delta
    return C, delta, rows.sum()


def run(tag, folds, members, which):
    pav, cnv, fs = genome_tables()
    sym = g64.sgd_symbol_to_orf()
    out = []
    for seed, held in folds:
        if tag == "val":
            fo = build_fold()
        else:
            base_meta = build_fold().meta
            fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
        proteins = [str(p) for p in fo.proteins]
        orfs = [sym.get(p.upper()) for p in proteins]
        coef, base, den = fit_coeffs(fo, proteins, orfs, pav, cnv, fs, held)
        C, delta, nrows = correction(fo, proteins, orfs, pav, cnv, fs, coef, held)
        acc = None
        for m in members:
            f = (os.path.join(RES, "pool_val", f"{m}.npy") if tag == "val"
                 else os.path.join(RES, "pool_real", f"{seed}_{held}__{m}.npy"))
            p = np.load(f).astype(np.float64)
            acc = p if acc is None else acc + p
        P0 = (acc / len(members)).astype(np.float32)
        print(f"[{tag}] seed{seed} {held}: coef={np.round(coef,3)} base={base:+.3f} "
              f"flagged proteins={int((np.abs(delta)>1e-6).sum())} rows={nrows}", flush=True)
        for k in SCALES:
            r = summary_row(f"k={k}", evaluate(fo, (P0 + k * C).astype(np.float32), which))
            r.update({"seed": seed, "strain": held, "k": k})
            out.append(r)
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    print("=== 六个无孤儿内层折（留出 CGD / BAH）===", flush=True)
    inner = run("inner", FOLDS, SHIPPED, INNER)
    inner.to_csv(os.path.join(RES, "genome_corr_inner.csv"), index=False)
    piv = inner.pivot_table(index=["seed", "strain"], columns="k", values="TOTAL")
    base = piv[0.0]
    summ = pd.DataFrame({"mean": piv.mean()})
    summ["delta"] = [(piv[c] - base).mean() for c in summ.index]
    summ["sem"] = [(piv[c] - base).sem() for c in summ.index]
    summ["up"] = [int(((piv[c] - base) > 0).sum()) for c in summ.index]
    print("\n" + summ.round(5).to_string())
    for mod in ["M2_rawFC(25%)", "M4_drug(20%)", "M1_abs(20%)", "FC[strain_only]", "FC[both]"]:
        if mod in inner.columns:
            q = inner.pivot_table(index=["seed", "strain"], columns="k", values=mod)
            print(f"  {mod:18s} " + "  ".join(f"k={c}:{(q[c]-q[0.0]).mean():+.5f}" for c in q.columns))

    print("\n=== 官方 val 镜像（留出菌株 BAI，只评一次）===", flush=True)
    val = run("val", [(0, "BAI")], SHIPPED, VAL)
    val.to_csv(os.path.join(RES, "genome_corr_val.csv"), index=False)
    cols = ["k", "TOTAL", "FC[chem_only]", "FC[strain_only]", "FC[both]", "FC[time]",
            "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)"]
    print(val[cols].round(4).to_string(index=False))

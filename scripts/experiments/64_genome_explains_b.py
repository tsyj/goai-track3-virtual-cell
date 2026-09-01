"""How much of the unseen-strain baseline b can the genome explain?  (train rows only)

docs/FINDINGS §2 measured the thing that dominates the unseen-strain splits: a
per-(strain, protein) baseline shift b, rms 0.34-0.36 log2, twice the size of the
true perturbation effect -- and concluded it is unestimable, because a held-out
strain has no training labels and its control wells are hidden.

That argument is about *labels*.  It says nothing about the genome.  All five
competition strains are isolates of the 1,011 Yeast Genomes Project, whose public
release gives, per isolate and per gene: presence/absence, copy number, homozygous
frameshift, and dN/dS.  None of that needs a single label from the strain.

This script measures the ceiling: fit b_hat on the four visible strains, and ask
what fraction of its variance genome features explain -- with a leave-one-strain-out
split, so the number is the honest out-of-strain R^2, not an in-sample fit.

    python scripts/64_genome_explains_b.py

Jiao Xinyuan 2026-09-02
"""
import gzip
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import protein_keep_mask                      # noqa: E402
from vcell.io import load_proteome                               # noqa: E402

GEN = os.path.join(ROOT, "data", "genomes")
OUT = os.path.join(ROOT, "results")
STRAINS = ["BAH", "BAI", "CEK", "CGD", "CRD"]


def sgd_symbol_to_orf():
    cols = ["sgdid", "type", "qual", "systematic", "standard", "alias", "parent",
            "sgdid2", "chrom", "start", "stop", "strand", "genpos", "cdate", "mdate", "desc"]
    df = pd.read_csv(os.path.join(GEN, "sgd", "SGD_features.tab"), sep="\t",
                     header=None, names=cols, dtype=str, quoting=3)
    df = df[df["type"] == "ORF"]
    m = {}
    for sys_, std, ali in zip(df["systematic"], df["standard"], df["alias"]):
        if not isinstance(sys_, str):
            continue
        m[sys_.upper()] = sys_
        if isinstance(std, str) and std:
            m[std.upper()] = sys_
        if isinstance(ali, str) and ali:
            for a in ali.split("|"):
                a = a.strip().upper()
                if a:
                    m.setdefault(a, sys_)
    return m


def _orf_of(col):
    tail = col.split(".", 1)[1] if "." in col else col
    tail = tail.split("_NumOfGenes")[0]
    return tail.upper() if len(tail) >= 7 and tail[0] == "Y" and tail[2] in "LR" else None


def strain_by_orf(path):
    """matrices whose ROWS are strains and COLUMNS are pangenome ORFs."""
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    df = df.loc[[s for s in STRAINS if s in df.index]]
    out = {}
    for c in df.columns:
        o = _orf_of(c)
        if o:
            out.setdefault(o, []).append(c)
    return pd.DataFrame({o: (df[cs].min(axis=1) if len(cs) > 1 else df[cs[0]])
                         for o, cs in out.items()})


def orf_by_strain(path, gene_col, id_cols):
    """matrices whose ROWS are genes and COLUMNS are strains (frameshift, dN/dS)."""
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", low_memory=False)
    have = [s for s in STRAINS if s in df.columns]
    g = df[gene_col].astype(str).str.upper().str.replace(r"\..*$", "", regex=True)
    sub = df[have].apply(pd.to_numeric, errors="coerce")
    sub.index = g
    sub = sub.groupby(level=0).max()
    return sub.T          # strains x ORF


if __name__ == "__main__":
    P = load_proteome("train_val")
    meta, Y = P.meta, P.X
    keep = protein_keep_mask(meta, Y)
    Y = Y[:, keep]
    proteins = [str(p) for p in P.proteins[keep]]
    tr = (meta["split_final"] == "train").to_numpy()
    sym = sgd_symbol_to_orf()
    orfs = [sym.get(p.upper()) for p in proteins]

    pav = strain_by_orf(os.path.join(GEN, "genesMatrix_PresenceAbsence.tab.gz"))
    cnv = strain_by_orf(os.path.join(GEN, "genesMatrix_CopyNumber.tab.gz"))
    try:
        fs = orf_by_strain(os.path.join(GEN, "genesMatrix_Frameshift.tab.gz"), "Gene", None)
    except Exception as e:
        print("frameshift load failed:", e); fs = None
    print(f"pav {pav.shape}  cnv {cnv.shape}  frameshift {None if fs is None else fs.shape}", flush=True)

    # ---- empirical per-(strain, protein) baseline on TRAIN rows only
    vis = [s for s in pav.index if ((meta["Strains"] == s) & tr).sum() > 0]
    rows = []
    for s in vis:
        rs = tr & (meta["Strains"] == s).to_numpy()
        ot = tr & (meta["Strains"] != s).to_numpy()
        ms = np.nanmean(np.where(rs[:, None], Y, np.nan), 0)
        mo = np.nanmean(np.where(ot[:, None], Y, np.nan), 0)
        fs_obs = np.isfinite(Y[rs]).mean(0)
        fo_obs = np.isfinite(Y[ot]).mean(0)
        for j, (p, o) in enumerate(zip(proteins, orfs)):
            if o is None or o not in pav.columns or not np.isfinite(pav.loc[s, o]):
                continue
            rows.append({
                "strain": s, "protein": p, "orf": o,
                "b": ms[j] - mo[j], "dobs": fs_obs[j] - fo_obs[j],
                "abund": mo[j], "obs_other": fo_obs[j],
                "pav": float(pav.loc[s, o]),
                "cnv": float(cnv.loc[s, o]) if o in cnv.columns else np.nan,
                "fshift": float(fs.loc[s, o]) if (fs is not None and s in fs.index and o in fs.columns) else np.nan,
            })
    d = pd.DataFrame(rows).dropna(subset=["b"])
    d["absent"] = (d.pav < 0.5).astype(float)
    d["fs"] = (d.fshift.fillna(0) > 0.5).astype(float)
    d["logcnv"] = np.log2(np.clip(d.cnv.fillna(1.0), 0.25, 8))
    print(f"\n(strain, protein) pairs: {len(d)}   strains: {sorted(d.strain.unique())}")
    print(f"b: rms {np.sqrt((d.b**2).mean()):.4f}  sd {d.b.std():.4f}")

    print("\n=== 单因子效应（相对该菌株之外的均值）===")
    for name, mask in [("基因缺失 PAV=0", d.absent > .5), ("纯合移码 frameshift", d.fs > .5),
                       ("拷贝数 <0.6", d.cnv < 0.6), ("拷贝数 >1.6", d.cnv > 1.6)]:
        m = mask.fillna(False)
        if m.sum():
            print(f"  {name:20s} n={int(m.sum()):5d}  b均值 {d.b[m].mean():+.4f}  "
                  f"中位 {d.b[m].median():+.4f}  检出率差 {d.dobs[m].mean():+.4f}")
    rest = ~((d.absent > .5) | (d.fs > .5))
    print(f"  {'其余（对照）':20s} n={int(rest.sum()):5d}  b均值 {d.b[rest].mean():+.4f}  "
          f"中位 {d.b[rest].median():+.4f}  检出率差 {d.dobs[rest].mean():+.4f}")

    # ---- leave-one-strain-out R^2 of genome features on b
    feats = ["absent", "fs", "logcnv"]
    X = d[feats].to_numpy(float)
    yb = d.b.to_numpy(float)
    print("\n=== 留一菌株外推：基因组特征对 b 的解释力 ===")
    tot_ss = tot_res = 0.0
    for s in sorted(d.strain.unique()):
        te = (d.strain == s).to_numpy()
        A = np.c_[np.ones(len(X)), X]
        w, *_ = np.linalg.lstsq(A[~te], yb[~te], rcond=None)
        pred = A[te] @ w
        ss = ((yb[te] - yb[~te].mean()) ** 2).sum()
        res = ((yb[te] - pred) ** 2).sum()
        tot_ss += ss; tot_res += res
        print(f"  留出 {s}: n={int(te.sum()):5d}  R2={1-res/ss:+.4f}  "
              f"rms(b)={np.sqrt((yb[te]**2).mean()):.4f} → 残差 rms={np.sqrt(res/te.sum()):.4f}")
    print(f"  合计 leave-one-strain-out R2 = {1-tot_res/tot_ss:+.4f}")

    # ---- ceiling if we only fixed the genes the genome flags
    flag = ((d.absent > .5) | (d.fs > .5)).to_numpy()
    print(f"\n基因组标记的 (菌株,蛋白) 对：{flag.sum()} / {len(d)} = {100*flag.mean():.2f}%")
    print(f"  这些对上 b 的方差占全部 b 方差的 {100*(yb[flag]**2).sum()/(yb**2).sum():.2f}%")
    print(f"  完美修正它们后 b 的 rms：{np.sqrt((yb**2).sum()/len(yb)):.4f} → "
          f"{np.sqrt((yb[~flag]**2).sum()/len(yb)):.4f}")

    d.to_csv(os.path.join(OUT, "genome_b_features.csv"), index=False)
    print("\nwrote results/genome_b_features.csv")

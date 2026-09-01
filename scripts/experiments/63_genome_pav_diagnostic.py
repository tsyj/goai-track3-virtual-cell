"""Does a strain's missing gene show up as a missing protein?  (train rows only)

The single largest unexplained error on the unseen-strain splits is a per-protein,
per-strain baseline shift b with rms 0.34-0.36 log2 -- larger than the true
perturbation effect (0.146).  docs/FINDINGS §2 concluded b is *unestimable*,
because a held-out strain has zero training labels and its own control wells are
hidden.  That conclusion assumed the only evidence about a strain is its labels.

It is not.  All five competition strains (CGD, BAH, BAI, CEK and the officially
held-out CRD) are isolates of the 1,011 Yeast Genomes Project (Peter et al.,
Nature 2018), which publishes gene presence/absence and copy number for 7,796
pangenome ORFs.  If a strain simply does not carry a gene, its protein cannot be
measured -- and that part of b is predictable with no labels at all.

This script tests the premise on TRAINING ROWS ONLY, so nothing here can leak:
for each visible strain, compare the measured log2 abundance of proteins the
genome says are absent in that strain against the same proteins in the strains
that do carry them.

Outputs results/genome_pav_diag.csv and prints the effect size.

    python scripts/63_genome_pav_diagnostic.py

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

# ---------------------------------------------------------------- gene names
def sgd_symbol_to_orf():
    """gene symbol (and aliases) -> systematic ORF name, from SGD_features.tab."""
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


def load_pav(path, strains):
    """rows = strain, cols = pangenome ORF -> DataFrame indexed by ORF systematic name."""
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    df = df.loc[[s for s in strains if s in df.index]]
    # column "X1768.YAL001C" / "X1771.YAL005C_NumOfGenes_3" -> YAL001C
    out = {}
    for c in df.columns:
        tail = c.split(".", 1)[1] if "." in c else c
        tail = tail.split("_NumOfGenes")[0]
        if len(tail) >= 7 and tail[0] == "Y" and tail[2] in "LR":
            out.setdefault(tail.upper(), []).append(c)
    keep = {orf: df[cs].min(axis=1) if len(cs) > 1 else df[cs[0]] for orf, cs in out.items()}
    return pd.DataFrame(keep)          # strains x ORF


if __name__ == "__main__":
    P = load_proteome("train_val")
    meta, Y = P.meta, P.X
    keep = protein_keep_mask(meta, Y)
    Y = Y[:, keep]
    proteins = [str(p) for p in P.proteins[keep]]
    tr = (meta["split_final"] == "train").to_numpy()
    print(f"train rows {tr.sum()}, proteins {len(proteins)}", flush=True)

    sym2orf = sgd_symbol_to_orf()
    orfs = [sym2orf.get(p.upper()) for p in proteins]
    mapped = sum(o is not None for o in orfs)
    print(f"protein symbol -> ORF mapped: {mapped}/{len(proteins)}", flush=True)

    strains = sorted(meta["Strains"].unique())
    pav = load_pav(os.path.join(GEN, "genesMatrix_PresenceAbsence.tab.gz"), strains)
    cnv = load_pav(os.path.join(GEN, "genesMatrix_CopyNumber.tab.gz"), strains)
    print(f"genome strains found: {list(pav.index)}  ORFs: {pav.shape[1]}", flush=True)

    rows = []
    for si, s in enumerate(pav.index):
        rs = tr & (meta["Strains"] == s).to_numpy()
        other = tr & (meta["Strains"] != s).to_numpy()
        if rs.sum() == 0:
            continue
        for j, (p, orf) in enumerate(zip(proteins, orfs)):
            if orf is None or orf not in pav.columns:
                continue
            present = pav.loc[s, orf]
            if not np.isfinite(present):
                continue
            ys, yo = Y[rs, j], Y[other, j]
            n_s, n_o = np.isfinite(ys).sum(), np.isfinite(yo).sum()
            rows.append({
                "strain": s, "protein": p, "orf": orf,
                "pav": int(round(float(present))), "cnv": float(cnv.loc[s, orf]) if orf in cnv.columns else np.nan,
                "n_obs_strain": int(n_s), "n_obs_other": int(n_o),
                "frac_obs_strain": n_s / max(rs.sum(), 1),
                "frac_obs_other": n_o / max(other.sum(), 1),
                "mean_strain": float(np.nanmean(ys)) if n_s else np.nan,
                "mean_other": float(np.nanmean(yo)) if n_o else np.nan,
            })
    d = pd.DataFrame(rows)
    d["delta"] = d["mean_strain"] - d["mean_other"]
    d["delta_frac_obs"] = d["frac_obs_strain"] - d["frac_obs_other"]
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "genome_pav_diag.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== 每菌株：基因组说“缺失”的蛋白数（在 4,422 个建模蛋白中）===")
    print(d.groupby("strain")["pav"].agg(n_mapped="size", n_absent=lambda x: int((x == 0).sum())).to_string())

    print("\n=== 缺失 vs 存在：该菌株相对其它菌株的 log2 偏移 (delta) 与检出率差 ===")
    g = d.groupby(["strain", "pav"]).agg(
        n=("delta", "size"),
        delta_mean=("delta", "mean"), delta_med=("delta", "median"),
        dfrac_mean=("delta_frac_obs", "mean"),
        frac_obs_strain=("frac_obs_strain", "mean")).round(4)
    print(g.to_string())

    ab = d[d.pav == 0].dropna(subset=["delta"])
    pr = d[d.pav == 1].dropna(subset=["delta"])
    print(f"\n全部菌株合并：缺失基因 n={len(ab)} delta={ab.delta.mean():+.4f} "
          f"(median {ab.delta.median():+.4f})  vs  存在基因 n={len(pr)} delta={pr.delta.mean():+.4f}")
    print(f"检出率：缺失 {ab.frac_obs_strain.mean():.3f} vs 该蛋白在其它菌株 {ab.frac_obs_other.mean():.3f}"
          f"　｜存在 {pr.frac_obs_strain.mean():.3f} vs {pr.frac_obs_other.mean():.3f}")

    print("\n=== 拷贝数与丰度（存在的基因，按 CNV 分档）===")
    q = d[(d.pav == 1) & d.cnv.notna()].copy()
    q["cnv_bin"] = pd.cut(q.cnv, [-0.1, 0.6, 1.4, 2.4, 99], labels=["<0.6", "~1", "~2", ">2.4"])
    print(q.groupby("cnv_bin")["delta"].agg(["size", "mean", "median"]).round(4).to_string())
    print("\nwrote results/genome_pav_diag.csv")

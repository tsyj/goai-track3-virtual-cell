"""Does mechanism-of-action class carry signal, once the source confound is removed?

Test: correlate every pair of compound effect vectors.  Compare same-MoA pairs
against different-MoA pairs, *within the same data_source* (WAYB and WAYC use
disjoint compound panels and different instruments, so cross-source pairs mix the
question with a batch effect).  Correlations are disattenuated by each
compound's split-half reliability, so a weak-but-real relation is not hidden by
estimation noise.
"""
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.chem import MOA, load_pubchem, fingerprints, tanimoto   # noqa: E402
from vcell.harness import build_fold                               # noqa: E402
from vcell.models import UnifiedBackfit                            # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)
rng = np.random.default_rng(0)

# coarser families, so classes actually have >1 member
FAMILY = {
    "translation_cytosolic": "translation", "aminoglycoside": "translation",
    "translation_mitochondrial": "translation_mito",
    "dna_damage_alkylating": "dna_damage", "dna_damage_crosslink": "dna_damage",
    "dna_damage_topoisomerase": "dna_damage", "replication_stress": "dna_damage",
    "ergosterol_azole": "ergosterol", "ergosterol_polyene": "ergosterol",
    "oxidative_thiol": "oxidative", "osmotic_ionic": "osmotic",
    "pi3k_tor": "tor_pi3k", "tor": "tor_pi3k",
}

f = build_fold(vehicle="both")
meta, obs = f.meta, f.obs_mask
treated = (~meta["is_control"] & ~meta["is_qc"]).to_numpy()
um = UnifiedBackfit(pert_factors=[]).fit(meta, f.Y_obs, obs)
R = np.where((obs & treated)[:, None], f.Y - um.predict(), np.nan).astype(np.float32)

comp = meta["compound"].to_numpy()
half = rng.integers(0, 2, len(meta))
recs = {}
for c in sorted(set(comp[obs & treated])):
    idx = np.where((comp == c) & obs & treated)[0]
    if len(idx) < 20 or c in ("Water", "DMSO", "Quality Control"):
        continue
    a = np.nanmean(R[idx[half[idx] == 0]], 0)
    b = np.nanmean(R[idx[half[idx] == 1]], 0)
    m = np.isfinite(a) & np.isfinite(b)
    r = np.corrcoef(a[m], b[m])[0, 1]
    recs[c] = {"vec": np.nanmean(R[idx], 0), "rel": 2 * r / (1 + r),
               "src": pd.Series(meta["data_source"].to_numpy()[idx]).mode()[0],
               "moa": FAMILY.get(MOA.get(c, "unknown"), MOA.get(c, "unknown"))}
names = list(recs)
print(f"{len(names)} compounds with a usable effect vector")
fam = pd.Series({c: recs[c]["moa"] for c in names})
print("\nfamilies with >1 member:")
print(fam.value_counts()[lambda s: s > 1].to_string())

chem = load_pubchem().set_index("name").reindex(names).reset_index()
F, _ = fingerprints(chem)
T = tanimoto(F)
lut = {c: i for i, c in enumerate(names)}

rows = []
for a, b in combinations(names, 2):
    va, vb = recs[a]["vec"], recs[b]["vec"]
    m = np.isfinite(va) & np.isfinite(vb)
    if m.sum() < 200:
        continue
    r = np.corrcoef(va[m], vb[m])[0, 1]
    denom = np.sqrt(max(recs[a]["rel"], 1e-3) * max(recs[b]["rel"], 1e-3))
    rows.append({"a": a, "b": b,
                 "same_source": recs[a]["src"].startswith("WAYB") ==
                                recs[b]["src"].startswith("WAYB"),
                 "same_moa": recs[a]["moa"] == recs[b]["moa"] != "unknown",
                 "tanimoto": T[lut[a], lut[b]],
                 "r": r, "r_disatt": np.clip(r / denom, -1, 1)})
P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, "compound_pair_similarity.csv"), index=False)

print("\n=== effect-vector correlation by MoA agreement ===")
for ss in (True, False):
    sub = P[P.same_source == ss]
    if not len(sub):
        continue
    g = sub.groupby("same_moa")[["r", "r_disatt"]].agg(["mean", "median", "count"])
    print(f"\nsame_source={ss}")
    print(g.to_string())
    a = sub[sub.same_moa].r_disatt.to_numpy()
    b = sub[~sub.same_moa].r_disatt.to_numpy()
    if len(a) > 1 and len(b) > 1:
        obs_d = a.mean() - b.mean()
        pool = np.r_[a, b]
        null = np.array([(lambda x: x[:len(a)].mean() - x[len(a):].mean())
                         (rng.permutation(pool)) for _ in range(20000)])
        print(f"  same-MoA minus different-MoA (disattenuated) = {obs_d:+.4f}"
              f"   permutation p = {(np.abs(null) >= abs(obs_d)).mean():.4f}"
              f"   (n_same={len(a)}, n_diff={len(b)})")

print("\n=== structural similarity vs response similarity (same source only) ===")
sub = P[P.same_source]
ok = np.isfinite(sub.tanimoto) & np.isfinite(sub.r_disatt)
print("  Pearson  corr(tanimoto, r_disatt) = %.3f"
      % np.corrcoef(sub.tanimoto[ok], sub.r_disatt[ok])[0, 1])
from scipy.stats import spearmanr  # noqa: E402
rho, pv = spearmanr(sub.tanimoto[ok], sub.r_disatt[ok])
print("  Spearman rho = %.3f  p = %.3g  (n=%d pairs)" % (rho, pv, ok.sum()))

print("\ntop-10 most similar response pairs (disattenuated), same source:")
print(sub.sort_values("r_disatt", ascending=False).head(10)
      [["a", "b", "same_moa", "tanimoto", "r", "r_disatt"]].to_string(index=False))

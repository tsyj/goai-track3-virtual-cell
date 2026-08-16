"""Can an unseen compound's proteome response be borrowed from chemical relatives?

Honest test: leave-one-compound-out over the *training* compounds only.  For each
held-out compound we re-estimate its effect vector purely from the others via the
chemical kernel, and correlate against its measured effect vector.  The val/test
compounds are never used to build or tune the kernel.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.chem import load_pubchem, similarity_matrix, transfer_weights  # noqa: E402
from vcell.harness import build_fold                                      # noqa: E402
from vcell.models import UnifiedBackfit, interaction_codes                # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)

f = build_fold(vehicle="both")
meta, obs = f.meta, f.obs_mask

print("fitting unified model to extract compound effect vectors ...")
um = UnifiedBackfit().fit(meta, f.Y_obs, obs)
codes, k = interaction_codes(meta, ["compound"])
comp_levels = (pd.Series(meta["compound"].astype(str).to_numpy())
               .groupby(codes).first().to_numpy())
A = um.terms["compound"]                       # (n_compounds, n_proteins)
seen = np.zeros(k, bool)
for c in np.unique(codes[obs & ~meta["is_control"].to_numpy()]):
    seen[c] = True
print(f"compound effect vectors: {k} levels, {seen.sum()} with visible data")

chem = load_pubchem().set_index("name").reindex(comp_levels).reset_index()
chem["moa"] = chem["moa"].fillna("unknown")
print("compounds without SMILES:", chem["smiles"].isna().sum())

role = meta.groupby("compound")["chemical_role"].first().reindex(comp_levels).to_numpy()
is_ctrl = np.array([c in ("Water", "DMSO", "Quality Control") for c in comp_levels])
donors_all = np.where(seen & ~is_ctrl)[0]

# effect magnitude, to weight the summary and to drop compounds that did nothing
mag = np.sqrt(np.nanmean(A ** 2, axis=1))


def loco(w_struct, w_moa, topk, temp):
    S = similarity_matrix(chem, w_struct, w_moa)
    W = transfer_weights(S, donors_all, donors_all, topk=topk, temp=temp)
    pred = W @ A[donors_all]
    r = []
    for i, t in enumerate(donors_all):
        a, b = pred[i], A[t]
        m = np.isfinite(a) & np.isfinite(b)
        r.append(np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 10 and a[m].std() > 0
                 else np.nan)
    return np.array(r), S


print("\n=== leave-one-compound-out transfer (training compounds only) ===")
grid = []
for w_s, w_m in [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]:
    for topk in (3, 5, 8):
        for temp in (0.1, 0.3):
            r, _ = loco(w_s, w_m, topk, temp)
            grid.append({"w_struct": w_s, "w_moa": w_m, "topk": topk, "temp": temp,
                         "mean_r": np.nanmean(r), "median_r": np.nanmedian(r),
                         "frac_pos": np.nanmean(r > 0),
                         "wmean_r": np.nansum(r * mag[donors_all]) /
                                    np.nansum(mag[donors_all] * np.isfinite(r))})
g = pd.DataFrame(grid).sort_values("wmean_r", ascending=False)
print(g.to_string(index=False))
g.to_csv(os.path.join(OUT, "chem_loco_grid.csv"), index=False)

best = g.iloc[0]
r, S = loco(best.w_struct, best.w_moa, int(best.topk), best.temp)
det = pd.DataFrame({"compound": comp_levels[donors_all], "moa": chem["moa"].to_numpy()[donors_all],
                    "effect_rms": mag[donors_all].round(3), "loco_r": r.round(3)})
print("\nper-compound transfer with the best kernel "
      f"(w_struct={best.w_struct}, w_moa={best.w_moa}, topk={int(best.topk)}, "
      f"temp={best.temp}):")
print(det.sort_values("effect_rms", ascending=False).to_string(index=False))
det.to_csv(os.path.join(OUT, "chem_loco_per_compound.csv"), index=False)

# what will the actually-held-out compounds borrow from?
S_best = similarity_matrix(chem, best.w_struct, best.w_moa)
held = np.where(~seen & ~is_ctrl)[0]
W = transfer_weights(S_best, held, donors_all, topk=int(best.topk), temp=best.temp)
print("\n=== donors for the compounds that are actually held out ===")
for i, t in enumerate(held):
    top = np.argsort(-W[i])[:3]
    src = ", ".join(f"{comp_levels[donors_all[j]]} ({W[i, j]:.2f})"
                    for j in top if W[i, j] > 0.01)
    print(f"  {comp_levels[t][:34]:36s} [{chem['moa'].to_numpy()[t]:24s}] <- {src or 'NOTHING'}")

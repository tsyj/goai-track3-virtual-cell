"""Adversarial review, track 2: does the model recover known yeast biology?

Everything so far has been scored with correlation metrics.  None of it shows
that the model learned biology rather than batch structure.  So: pre-specified,
falsifiable predictions from textbook S. cerevisiae responses, checked first in
the measured data (is the effect even there?) and then in the model's prediction
(did the model recover it?).

Marker sets and expected directions are fixed before looking at any result.
A miss is reported as a miss.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold                                      # noqa: E402
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,   # noqa: E402
                          UnifiedBackfit)

from scipy.stats import mannwhitneyu                                      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 280)
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}

# --- pre-registered marker sets -------------------------------------------
EXPLICIT = {
    "heat_shock": ["HSP12", "HSP26", "HSP42", "HSP78", "HSP82", "HSC82", "HSP104",
                   "SSA1", "SSA2", "SSA3", "SSA4", "STI1", "SIS1"],
    "oxidative": ["CTT1", "TSA1", "TSA2", "GPX1", "GPX2", "TRX2", "SOD1", "SOD2",
                  "AHP1", "CCP1", "GRX1", "GRX2", "PRX1"],
    "upr_er": ["KAR2", "PDI1", "ERO1", "LHS1", "SCJ1", "EUG1", "MPD1", "JEM1"],
    "rnr_dna": ["RNR1", "RNR2", "RNR3", "RNR4", "DDR48", "HUG1", "RAD51", "RAD54"],
    "aa_biosynth": ["ILV1", "ILV2", "ILV3", "ILV5", "ILV6", "LEU1", "LEU4", "LEU9",
                    "ARG1", "ARG3", "ARG4", "HIS4", "TRP2", "TRP3", "TRP4", "TRP5"],
    "osmotic": ["GPD1", "GPP2", "HOR2", "ALD3", "TPS1", "TPS2", "TSL1", "NTH1"],
}
# expectation: (compound, marker set, expected direction)
PREDICTIONS = [
    ("Rapamycin", "ribosome", "down"),      # TORC1 inhibition shuts ribosome synthesis
    ("Rapamycin", "heat_shock", "up"),      # ...and de-represses the stress programme
    ("Tunicamycin", "upr_er", "up"),        # blocks N-glycosylation -> UPR
    ("Geldanamycin", "heat_shock", "up"),   # Hsp90 inhibition -> heat-shock response
    ("Hydroxyurea", "rnr_dna", "up"),       # RNR inhibition -> RNR induction
    ("Sulfometuron methyl", "aa_biosynth", "up"),   # Ilv2 block -> Gcn4 programme
    ("NaCl", "osmotic", "up"),
    ("Sorbitol", "osmotic", "up"),
    ("Cisplatin", "rnr_dna", "up"),
    ("Parthenolide", "oxidative", "up"),    # thiol-reactive -> oxidative programme
]

f = build_fold(vehicle="both")
meta, prot = f.meta, np.array([str(p) for p in f.proteins])
sets = {k: np.isin(prot, v) for k, v in EXPLICIT.items()}
sets["ribosome"] = np.array([p.startswith(("RPL", "RPS")) for p in prot])
for k, v in sets.items():
    print(f"  marker set {k:14s} {v.sum():4d} proteins present")

um = UnifiedBackfit(
    batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
    pert_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
).fit(meta, f.Y_obs, f.obs_mask)
P = um.predict()
rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03, n_jobs=32)
rb.fit(meta, f.Y_obs, f.obs_mask, P)
P = P + rb.predict()

D_true = np.where(np.isfinite(f.C_true), f.Y - f.C_true, np.nan)
D_pred = P - f.C_true
treated = (~meta["is_control"] & ~meta["is_qc"]).to_numpy()
# late time points, where a proteome-level response has had time to appear
late = treated & (meta["pert_time"] >= 120).to_numpy()


def enrichment(D, rows, mask):
    """Median marker Delta minus median of all other proteins, + rank-test p."""
    d = np.nanmean(D[rows], axis=0)
    a, b = d[mask & np.isfinite(d)], d[~mask & np.isfinite(d)]
    if len(a) < 4:
        return np.nan, np.nan
    return float(np.median(a) - np.median(b)), float(
        mannwhitneyu(a, b, alternative="two-sided").pvalue)


rows_out = []
print("\n=== pre-registered predictions (samples at >= 120 min) ===")
print(f"{'compound':22s} {'marker set':13s} {'exp':5s} {'seen?':6s} "
      f"{'measured':>9s} {'p':>9s}  {'model':>9s} {'p':>9s}  verdict")
for comp, mset, direction in PREDICTIONS:
    rows = np.where(late & (meta["compound"] == comp).to_numpy())[0]
    if len(rows) == 0:
        print(f"{comp:22s} {mset:13s} -- no samples")
        continue
    m_eff, m_p = enrichment(D_true, rows, sets[mset])
    p_eff, p_p = enrichment(D_pred, rows, sets[mset])
    want = 1 if direction == "up" else -1
    meas_ok = np.sign(m_eff) == want and m_p < 0.05
    mod_ok = np.sign(p_eff) == want and p_p < 0.05
    verdict = ("data+model" if meas_ok and mod_ok else
               "data only" if meas_ok else
               "model only" if mod_ok else "neither")
    role = meta.loc[meta["compound"] == comp, "chemical_role"].iloc[0]
    rows_out.append({"compound": comp, "markers": mset, "expected": direction,
                     "chemical_role": role, "visible_to_model": role == "train",
                     "n_samples": len(rows), "measured_effect": m_eff,
                     "measured_p": m_p, "model_effect": p_eff, "model_p": p_p,
                     "verdict": verdict})
    print(f"{comp:22s} {mset:13s} {direction:5s} {role:6s} {m_eff:+9.3f} {m_p:9.1e}  "
          f"{p_eff:+9.3f} {p_p:9.1e}  {verdict}")

df = pd.DataFrame(rows_out)
df.to_csv(os.path.join(OUT, "biology_check.csv"), index=False)
n = len(df)
print(f"\nconfirmed in the measured data: {(df.verdict.isin(['data+model','data only'])).sum()}/{n}")
print(f"also recovered by the model    : {(df.verdict == 'data+model').sum()}/{n}")
real = df[df.verdict.isin(["data+model", "data only"])]
seen = real[real.visible_to_model]
held = real[~real.visible_to_model]
print(f"\n  of the {len(seen)} real effects whose compound the model can SEE:      "
      f"{(seen.verdict == 'data+model').sum()} recovered")
print(f"  of the {len(held)} real effects whose compound is HELD OUT:      "
      f"{(held.verdict == 'data+model').sum()} recovered   "
      f"({', '.join(held.compound)})")
print("\n(a prediction that fails in the measured data is a fact about the "
      "experiment,\n not about the model -- those rows are reported, not dropped.)")

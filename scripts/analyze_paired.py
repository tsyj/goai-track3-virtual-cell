"""Paired report for any of the six-fold searches, split by orphan-plate exposure.

Three of the six inner mirrors put 32% of their evaluation rows on plates that
carry no training label at all, because holding out a strain can empty a whole
plate.  The official val split and the official test set contain **zero** such
rows (checked over all 4,454 test rows, including all 2,231 CRD rows).  Any effect
that acts through the plate term therefore gets scored on the inner mirrors in a
regime that does not exist in the real evaluation -- the instrument level looked
like +0.0071 that way and is +0.0016 once restricted.

So every comparison gets reported twice: all six folds, and the three folds whose
structure matches the official protocol.  Adoption decisions use the latter.

    python scripts/analyze_paired.py results/instrument_level_raw.csv "REF (no instrument)"

Jiao Xinyuan 2026-08-16
"""
import sys

import numpy as np
import pandas as pd

ORPHAN_FREE = [(1, "CGD"), (3, "BAH"), (5, "CGD")]      # 0% orphan-plate rows
ORPHAN = [(0, "CEK"), (2, "DHY210"), (4, "CEK")]        # ~32% orphan-plate rows

path = sys.argv[1]
raw = pd.read_csv(path)
ref = sys.argv[2] if len(sys.argv) > 2 else None
value = sys.argv[3] if len(sys.argv) > 3 else "TOTAL"

piv = raw.pivot_table(index=["seed", "strain"], columns="config", values=value)
if ref is None:
    ref = piv.mean().idxmin()          # fall back to the weakest as reference
cur = piv[ref]

rows = []
for c in piv.columns:
    d = piv[c] - cur
    have_free = [k for k in ORPHAN_FREE if k in d.index]
    have_orph = [k for k in ORPHAN if k in d.index]
    rows.append({
        "config": c,
        "all6_mean": piv[c].mean(),
        "all6_delta": d.mean(), "all6_sem": d.sem(),
        "orphan_delta": d.loc[have_orph].mean() if have_orph else np.nan,
        "free_delta": d.loc[have_free].mean() if have_free else np.nan,
        "free_sem": d.loc[have_free].sem() if len(have_free) > 1 else np.nan,
        "free_up": int((d.loc[have_free] > 0).sum()) if have_free else 0,
        "free_n": len(have_free),
    })
out = pd.DataFrame(rows).sort_values("free_delta", ascending=False)
pd.set_option("display.width", 240)
print(f"{path}   value={value}   ref='{ref}'")
print("orphan_delta = 三个含孤儿 plate 的折; free_delta = 三个与官方结构一致的折（决策看这一列）")
print(out.round(5).to_string(index=False))

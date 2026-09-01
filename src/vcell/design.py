"""Experimental-design utilities: control matching, context keys, split mirrors.

The 96-well design (recovered from the metadata, see docs/EDA.md):

* one ``Yeast_cell_plate`` == one (data_source, Medium, Temperature, pert_time,
  instrument) cell.  Plate identity therefore *is* the measurement context.
* WAYB plates: 6 strains x 16 wells-conditions (Water, DMSO + 13 compounds + an
  EDTA duplicate).  WAYC plates: 2 strains x 48 conditions.
* Every plate that carries test samples also carries labelled train samples, so
  plate-level effects are estimable without ever touching test labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import CONTROL_KEYS, CONTROL_COMPOUNDS

# Which vehicle each compound was dissolved in.  Unknown for this dataset -- the
# handbook only says "matched to DMSO or Water by the official rule".  We keep it
# configurable and run a sensitivity analysis (scripts/06_vehicle_sensitivity.py).
VEHICLE_STRATEGIES = ("dmso", "water", "both", "curated")

# Curated solubility assignment (public chemistry knowledge -> open-knowledge board).
WATER_SOLUBLE = {
    "EDTA", "NaCl", "Sorbitol", "H2O2", "Hydroxyurea", "MMS", "SDS", "G418",
    "Hygromycin B", "Neomycin B", "Cisplatin", "CHX", "Doxycycline hyclate",
}


def vehicle_for(compound: str, strategy: str = "dmso") -> tuple:
    """Return the tuple of acceptable control compounds, in preference order."""
    if strategy == "dmso":
        return ("DMSO", "Water")
    if strategy == "water":
        return ("Water", "DMSO")
    if strategy == "both":
        return CONTROL_COMPOUNDS
    if strategy == "curated":
        return ("Water", "DMSO") if compound in WATER_SOLUBLE else ("DMSO", "Water")
    raise ValueError(strategy)


def build_control_index(meta: pd.DataFrame) -> dict:
    """ctx_key -> {control_compound: [row positions]} using *row order of meta*."""
    idx: dict[str, dict[str, list[int]]] = {}
    ctrl = meta.index[meta["is_control"].to_numpy()]
    for pos in ctrl:
        r = meta.iloc[pos] if not isinstance(pos, (int, np.integer)) else meta.iloc[pos]
        idx.setdefault(r["ctx_key"], {}).setdefault(r["compound"], []).append(int(pos))
    return idx


def match_controls(meta: pd.DataFrame, strategy: str = "both") -> pd.Series:
    """For each row, the list of row-positions of its matched control samples.

    Matching keys are exactly the handbook's: data_source, strain, medium,
    temperature, time, instrument, plate.  Control rows map to themselves-excluded
    (a control is not its own control).
    """
    meta = meta.reset_index(drop=True)
    ctx_ctrl: dict[tuple, list[int]] = {}
    for (ctx, comp), grp in meta[meta["is_control"]].groupby(["ctx_key", "compound"]):
        ctx_ctrl[(ctx, comp)] = grp.index.tolist()

    out = []
    for pos, (ctx, comp, isctrl) in enumerate(
        zip(meta["ctx_key"], meta["compound"], meta["is_control"])
    ):
        prefs = vehicle_for(comp, strategy) if not isctrl else (comp,)
        picked: list[int] = []
        if strategy == "both" and not isctrl:
            for c in CONTROL_COMPOUNDS:
                picked += ctx_ctrl.get((ctx, c), [])
        else:
            for c in prefs:
                got = ctx_ctrl.get((ctx, c), [])
                if got:
                    picked = list(got)
                    break
        if isctrl:
            picked = [p for p in picked if p != pos]
        out.append(picked)
    return pd.Series(out, index=meta.index, name="control_rows")


def control_reference(X: np.ndarray, control_rows: pd.Series) -> np.ndarray:
    """(n_samples, n_proteins) mean log2 control profile, NaN where unavailable."""
    n, p = X.shape
    ref = np.full((n, p), np.nan, dtype=np.float32)
    for i, rows in enumerate(control_rows):
        if not rows:
            continue
        block = X[rows]
        with np.errstate(invalid="ignore"):
            ref[i] = np.nanmean(block, axis=0)
    return ref


# ---------------------------------------------------------------------------
# split mirrors: the organisers already shipped a validation set whose four
# val_* splits mirror the four test_* splits one-for-one.
MIRROR = {
    "val_chem_only": "test_chem_only",     # S1  unseen compound, seen strain
    "val_strain_only": "test_strain_only", # S2  unseen strain,  seen compound
    "val_both": "test_both",               # S3  both unseen
    "val_time": "test_time",               # time extrapolation
}


def frozen_train_mask(meta: pd.DataFrame) -> np.ndarray:
    """Rows usable for freezing normalisation / reference statistics."""
    return (meta["split_final"] == "train").to_numpy()

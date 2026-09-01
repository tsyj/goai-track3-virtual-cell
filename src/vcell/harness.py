"""Evaluation harness: build the local mirror of the official test protocol.

Only ``split_final == 'train'`` labels are ever visible to a model.  The four
``val_*`` splits shipped by the organisers mirror the four ``test_*`` splits
one-for-one, including the fact that the held-out strain's own control wells are
*also* hidden (BAI locally <-> CRD officially).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .design import match_controls, control_reference
from .io import load_proteome
from .metrics import ScoreConfig, Scorer


@dataclass
class Fold:
    meta: pd.DataFrame
    Y: np.ndarray            # (n, p) truth -- ORGANISER SIDE, never given to a model
    Y_obs: np.ndarray        # (n, p) truth with every non-train row set to NaN
    obs_mask: np.ndarray     # rows a model may learn from
    C_true: np.ndarray       # organiser-side matched-control reference
    C_obs: np.ndarray        # matched-control reference computable from Y_obs only
    control_rows: pd.Series
    scorer: Scorer
    proteins: np.ndarray

    @property
    def n(self) -> int:
        return self.Y.shape[0]


_CACHE: dict = {}


# Official protein filter (Guomics interpretation deck, section "Data
# preprocessing"): drop proteins missing in >= 80% of the *train* rows, computed
# on train rows only.  The deck's code says `missing_rate < 0.80`, which gives
# 4,422 proteins here, while the same deck states the result is 4,232 (that count
# corresponds to a ~0.70 threshold).  We implement the documented rule and expose
# the threshold; see docs/OPEN_QUESTIONS.md P0-7.
PROTEIN_MISSING_MAX = 0.80


def protein_keep_mask(meta: pd.DataFrame, Y: np.ndarray,
                      thresh: float = PROTEIN_MISSING_MAX) -> np.ndarray:
    train = (meta["split_final"] == "train").to_numpy()
    return np.isnan(Y[train]).mean(0) < thresh


def build_fold(vehicle: str = "both", cfg: ScoreConfig | None = None,
               splits: pd.Series | None = None,
               filter_proteins: bool = True,
               protein_thresh: float = PROTEIN_MISSING_MAX) -> Fold:
    """Assemble an evaluation fold.

    ``splits`` overrides ``split_final`` so a *second, inner* mirror can be built
    out of the training rows alone.  Hyper-parameters are tuned on the inner
    mirror; the organisers' val_* mirror is then scored once, untouched.

    ``filter_proteins`` applies the official low-coverage protein filter.  It is
    computed from the fold's own train rows, so an inner mirror filters on its
    own training set rather than inheriting the outer one.
    """
    key = ("proteome", vehicle)
    if key not in _CACHE:
        P = load_proteome("train_val")
        _CACHE[key] = (P, match_controls(P.meta.reset_index(drop=True),
                                         strategy=vehicle))
    P, control_rows = _CACHE[key]
    meta = P.meta.reset_index(drop=True).copy()
    Y = P.X
    if splits is not None:
        meta["split_final"] = np.asarray(splits)

    if filter_proteins:
        keep = protein_keep_mask(meta, Y, protein_thresh)
        Y = Y[:, keep]
        proteins = P.proteins[keep]
    else:
        proteins = P.proteins

    C_true = control_reference(Y, control_rows)
    obs = (meta["split_final"] == "train").to_numpy()
    Y_obs = np.where(obs[:, None], Y, np.nan).astype(np.float32)
    C_obs = control_reference(Y_obs, control_rows)

    eval_masks = {s: (meta["split_final"] == s).to_numpy()
                  for s in meta["split_final"].unique()}
    scorer = Scorer(meta, Y, C_true, obs, eval_masks, cfg,
                    control_rows=control_rows)
    return Fold(meta=meta, Y=Y, Y_obs=Y_obs, obs_mask=obs, C_true=C_true,
                C_obs=C_obs, control_rows=control_rows, scorer=scorer,
                proteins=proteins)


VAL = dict(s1="val_chem_only", s2="val_strain_only", s3="val_both", stime="val_time")
INNER = dict(s1="in_chem_only", s2="in_strain_only", s3="in_both", stime="in_time")


def make_inner_splits(meta: pd.DataFrame, hold_strain: str = "CEK",
                      n_hold_compounds: int = 8, seed: int = 0) -> pd.Series:
    """Carve an S1/S2/S3/time mirror out of the ``train`` rows only.

    Same construction rule as the organisers': hold out one strain entirely
    (including its control wells) and a set of compounds entirely, then a
    scattered time hold-out among the rest.
    """
    rng = np.random.default_rng(seed)
    s = pd.Series(np.where(meta["split_final"].to_numpy() == "train", "train", "unused"),
                  index=meta.index)
    tr = s == "train"

    pool = (meta.loc[tr & ~meta["is_control"] & ~meta["is_qc"]]
            .groupby(["data_source", "compound"]).size().reset_index())
    pool["wayb"] = pool["data_source"].str.startswith("WAYB")
    hold_c = []
    for wayb, grp in pool.groupby("wayb"):
        cands = sorted(set(grp["compound"]))
        k = max(1, int(round(n_hold_compounds * (0.3 if wayb else 0.7))))
        hold_c += list(rng.choice(cands, size=min(k, len(cands)), replace=False))
    hold_c = set(hold_c)

    bad_strain = tr & (meta["Strains"] == hold_strain).to_numpy()
    bad_chem = tr & meta["compound"].isin(hold_c).to_numpy()
    s[bad_strain & bad_chem] = "in_both"
    s[bad_strain & ~bad_chem] = "in_strain_only"
    s[~bad_strain & bad_chem] = "in_chem_only"

    rest = np.where((s == "train").to_numpy() & (~meta["is_control"]).to_numpy())[0]
    s.iloc[rng.choice(rest, size=min(160, len(rest)), replace=False)] = "in_time"
    return s


def evaluate(fold: Fold, P: np.ndarray, which: dict | None = None) -> dict:
    return fold.scorer.report(P, **(which or VAL))


def _strip_prefix(k: str) -> str:
    for p in ("val_", "in_", "test_"):
        if k.startswith(p):
            return k[len(p):]
    return k


def summary_row(name: str, rep: dict) -> dict:
    ps = {f"FC[{_strip_prefix(k)}]": v
          for k, v in rep.get("per_split_rawFC", {}).items()}
    return {
        "model": name,
        "TOTAL": rep["TOTAL"],
        **ps,
        "M1_abs(20%)": rep["M1_absolute"]["score"],
        "M2_rawFC(25%)": rep["M2_rawFC"]["score"],
        "M3_ctx(20%)": rep["M3_ctx_resid"]["score"],
        "M4_drug(20%)": rep["M4_drug_resid"]["score"],
        "M5_bt(10%)": rep["M5_both_time"]["score"],
        "M6_DEP(5%)": rep["M6_high_effect"]["score"],
        "sampPCC": rep["M1_absolute"]["sample_pcc"],
        "sampR2": rep["M1_absolute"]["sample_r2"],
        "protPCC": rep["M1_absolute"]["protein_pcc"],
        "dirAcc": rep["M6_high_effect"]["direction_acc"],
    }

"""Faithful re-implementation of the six official scoring modules.

Handbook (bf747305..., section 02 / 方向一) weights
--------------------------------------------------
  M1 absolute fidelity          20%   all splits   sample-wise & protein-wise PCC / R^2
  M2 matched-control raw FC     25%   all OOD      PCC(dPred, dTrue)
  M3 context-mean residual      20%   S1           PCC(dPred - mu_ctx, dTrue - mu_ctx)
  M4 drug-mean residual         20%   S2           PCC(dPred - mu_drug, dTrue - mu_drug)
  M5 both-unseen / time         10%   S3 + time    raw FC + absolute fidelity
  M6 high-effect / DEP           5%   all          direction acc, high-effect PCC, F1/AUPRC

All frozen statistics (mu_ctx, mu_drug) are computed from *training* rows only,
as the handbook requires ("所有参照、归一化统计与对照匹配规则须仅用训练数据冻结").

Unknowns w.r.t. the official implementation are exposed as arguments and swept in
scripts/09_metric_audit.py rather than silently hard-coded:
  * which vehicle (DMSO / Water) a compound is matched to,
  * whether multiple matched controls are averaged or one is picked,
  * the exact context key used for mu_ctx.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# low-level vectorised correlation / R^2 with NaN masking


def _pcc_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation over finite pairs.  NaN row -> NaN."""
    M = np.isfinite(A) & np.isfinite(B)
    A = np.where(M, A, 0.0).astype(np.float64)
    B = np.where(M, B, 0.0).astype(np.float64)
    n = M.sum(1)
    sa, sb = A.sum(1), B.sum(1)
    saa, sbb, sab = (A * A).sum(1), (B * B).sum(1), (A * B).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab / n - (sa / n) * (sb / n)
        va = saa / n - (sa / n) ** 2
        vb = sbb / n - (sb / n) ** 2
        r = cov / np.sqrt(va * vb)
    r[(n < 3) | ~np.isfinite(r)] = np.nan
    return r


def _r2_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Row-wise coefficient of determination 1 - SSres/SStot (can be negative)."""
    M = np.isfinite(pred) & np.isfinite(true)
    P = np.where(M, pred, 0.0).astype(np.float64)
    T = np.where(M, true, 0.0).astype(np.float64)
    n = M.sum(1)
    res = ((P - T) ** 2 * M).sum(1)
    mu = T.sum(1) / np.maximum(n, 1)
    tot = ((T - mu[:, None]) ** 2 * M).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - res / tot
    r2[(n < 3) | (tot <= 0) | ~np.isfinite(r2)] = np.nan
    return r2


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ---------------------------------------------------------------------------


@dataclass
class ScoreConfig:
    weights: dict = field(default_factory=lambda: {
        "M1_absolute": 0.20,
        "M2_rawFC": 0.25,
        "M3_ctx_resid": 0.20,
        "M4_drug_resid": 0.20,
        "M5_both_time": 0.10,
        "M6_high_effect": 0.05,
    })
    # context key for mu_ctx: shared drug response within one biological+batch cell
    ctx_cols: tuple = ("data_source", "Strains", "Medium", "Temperature",
                       "pert_time", "Yeast_cell_plate")
    # context key for mu_drug: average of a drug's effect over *training* contexts
    drug_col: str = "compound"
    dep_threshold: float = 1.0
    topk: int = 100
    # The handbook writes "Delta_pred = y_hat_treat - y_control" -- an un-hatted
    # y_control, i.e. the organiser's *measured* control on both sides.  It is
    # ambiguous whether the submitted prediction for the control sample is used
    # instead.  'measured' follows the notation; 'predicted' is the alternative.
    delta_mode: str = "measured"
    # the handbook lists "corr / R^2" for M1 without giving the aggregation.
    # 'pcc_only'  -> mean(sample PCC, protein PCC)
    # 'with_r2'   -> mean of all four, R^2 clipped at 0 (it is unbounded below)
    m1_aggregate: str = "with_r2"


def _average_precision(score: np.ndarray, label: np.ndarray) -> float:
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    order = np.argsort(-score, kind="stable")
    lab = label[order]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / lab.sum())


class Scorer:
    """Scores predictions on one evaluation fold.

    Parameters
    ----------
    meta          full metadata frame (row-aligned with ``Y`` and ``control_ref``)
    Y             (n, p) ground-truth log2 proteome, NaN = not measured
    control_ref   (n, p) matched-control log2 profile (mean of matched controls)
    train_mask    rows whose labels are allowed to be used for frozen statistics
    eval_masks    dict split-name -> boolean mask of rows to score
    """

    def __init__(self, meta: pd.DataFrame, Y: np.ndarray, control_ref: np.ndarray,
                 train_mask: np.ndarray, eval_masks: dict, cfg: ScoreConfig | None = None,
                 control_rows=None):
        self.meta = meta.reset_index(drop=True)
        self.Y = Y
        self.C = control_ref
        self.train_mask = np.asarray(train_mask)
        self.eval_masks = eval_masks
        self.cfg = cfg or ScoreConfig()
        self.control_rows = control_rows
        self.D_true = Y - control_ref
        self._freeze()

    def _pred_control(self, P: np.ndarray) -> np.ndarray:
        """The control reference a model would build from its own predictions."""
        if self.cfg.delta_mode == "measured" or self.control_rows is None:
            return self.C
        out = np.full_like(P, np.nan)
        for i, rows in enumerate(self.control_rows):
            if rows:
                out[i] = P[rows].mean(0)
        return out

    # -- frozen references -------------------------------------------------
    def _freeze(self) -> None:
        m, cfg = self.meta, self.cfg
        tr = self.train_mask & ~m["is_control"].to_numpy() & ~m["is_qc"].to_numpy()

        ctx = m[list(cfg.ctx_cols)].astype(str).agg("|".join, axis=1)
        self._ctx = ctx.to_numpy()
        self.mu_ctx = self._group_mean(self.D_true, self._ctx, tr)

        drug = m[cfg.drug_col].astype(str).to_numpy()
        self._drug = drug
        # mu_drug uses only *training strains/contexts* for that drug
        self.mu_drug = self._group_mean(self.D_true, drug, tr)

    @staticmethod
    def _group_mean(D: np.ndarray, keys: np.ndarray, use: np.ndarray) -> dict:
        out = {}
        idx = np.where(use)[0]
        order = np.argsort(keys[idx], kind="stable")
        idx = idx[order]
        ks = keys[idx]
        bounds = np.r_[0, np.where(ks[1:] != ks[:-1])[0] + 1, len(ks)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            rows = idx[a:b]
            with np.errstate(invalid="ignore"):
                out[ks[a]] = np.nanmean(D[rows], axis=0)
        return out

    def _lookup(self, table: dict, keys: np.ndarray, rows: np.ndarray) -> np.ndarray:
        p = self.Y.shape[1]
        nan = np.full(p, np.nan, dtype=np.float32)
        return np.stack([table.get(keys[r], nan) for r in rows])

    # -- modules -----------------------------------------------------------
    def _sel(self, names) -> np.ndarray:
        mask = np.zeros(len(self.meta), bool)
        for n in np.atleast_1d(names):
            mask |= self.eval_masks[n]
        return np.where(mask)[0]

    def m1_absolute(self, P: np.ndarray, rows: np.ndarray) -> dict:
        pcc_s = _pcc_rows(P[rows], self.Y[rows])
        r2_s = _r2_rows(P[rows], self.Y[rows])
        pcc_p = _pcc_rows(P[rows].T, self.Y[rows].T)
        r2_p = _r2_rows(P[rows].T, self.Y[rows].T)
        parts = [_nanmean(pcc_s), _nanmean(pcc_p)]
        if self.cfg.m1_aggregate == "with_r2":
            parts += [max(0.0, _nanmean(r2_s)), max(0.0, _nanmean(r2_p))]
        return {"sample_pcc": _nanmean(pcc_s), "sample_r2": _nanmean(r2_s),
                "protein_pcc": _nanmean(pcc_p), "protein_r2": _nanmean(r2_p),
                "score": float(np.mean(parts))}

    def m2_rawfc(self, P: np.ndarray, rows: np.ndarray) -> dict:
        dP = P[rows] - self._pred_control(P)[rows]
        r = _pcc_rows(dP, self.D_true[rows])
        return {"delta_pcc": _nanmean(r), "n": len(rows), "score": _nanmean(r)}

    def _residual(self, P: np.ndarray, rows: np.ndarray, table: dict,
                  keys: np.ndarray) -> dict:
        mu = self._lookup(table, keys, rows)
        dP = P[rows] - self._pred_control(P)[rows] - mu
        dT = self.D_true[rows] - mu
        r = _pcc_rows(dP, dT)
        return {"resid_pcc": _nanmean(r), "n_with_ref": int(np.isfinite(mu).any(1).sum()),
                "score": _nanmean(r)}

    def m3_ctx_resid(self, P: np.ndarray, rows: np.ndarray) -> dict:
        return self._residual(P, rows, self.mu_ctx, self._ctx)

    def m4_drug_resid(self, P: np.ndarray, rows: np.ndarray) -> dict:
        return self._residual(P, rows, self.mu_drug, self._drug)

    def m6_high_effect(self, P: np.ndarray, rows: np.ndarray) -> dict:
        thr, K = self.cfg.dep_threshold, self.cfg.topk
        dP = P[rows] - self._pred_control(P)[rows]
        dT = self.D_true[rows]
        hi = np.isfinite(dT) & (np.abs(dT) > thr) & np.isfinite(dP)
        with np.errstate(invalid="ignore"):
            agree = (np.sign(dP) == np.sign(dT)) & hi
        n_hi = hi.sum(1)
        dir_acc = np.where(n_hi > 0, agree.sum(1) / np.maximum(n_hi, 1), np.nan)
        hi_pcc = _pcc_rows(np.where(hi, dP, np.nan), np.where(hi, dT, np.nan))

        prec, rec, f1, ap = [], [], [], []
        for i in range(len(rows)):
            t = np.where(hi[i])[0]
            if len(t) == 0:
                continue
            v = np.abs(np.where(np.isfinite(dP[i]), dP[i], -np.inf))
            k = min(K, int(np.isfinite(dP[i]).sum()))
            pred = set(np.argpartition(-v, k - 1)[:k].tolist())
            tp = len(pred & set(t.tolist()))
            pr, rc = tp / max(k, 1), tp / len(t)
            prec.append(pr); rec.append(rc)
            f1.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
            lab = np.zeros(dP.shape[1]); lab[t] = 1
            keep = np.isfinite(dT[i]) & np.isfinite(dP[i])
            ap.append(_average_precision(np.abs(dP[i])[keep], lab[keep]))
        return {"direction_acc": _nanmean(dir_acc), "high_effect_pcc": _nanmean(hi_pcc),
                f"precision@{K}": _nanmean(prec), f"recall@{K}": _nanmean(rec),
                f"f1@{K}": _nanmean(f1), "AP": _nanmean(ap),
                "score": float(np.mean([_nanmean(dir_acc), _nanmean(hi_pcc),
                                        _nanmean(ap)]))}

    # -- full report -------------------------------------------------------
    def report(self, P: np.ndarray, s1: str, s2: str, s3: str, stime: str) -> dict:
        all_ood = self._sel([s1, s2, s3, stime])
        out = {
            "M1_absolute": self.m1_absolute(P, all_ood),
            "M2_rawFC": self.m2_rawfc(P, all_ood),
            "M3_ctx_resid": self.m3_ctx_resid(P, self._sel(s1)),
            "M4_drug_resid": self.m4_drug_resid(P, self._sel(s2)),
            "M6_high_effect": self.m6_high_effect(P, all_ood),
        }
        both, tm = self._sel(s3), self._sel(stime)
        out["M5_both_time"] = {
            "both_rawFC": self.m2_rawfc(P, both)["score"],
            "both_abs": self.m1_absolute(P, both)["score"],
            "time_abs": self.m1_absolute(P, tm)["score"],
            "time_rawFC": self.m2_rawfc(P, tm)["score"],
        }
        out["M5_both_time"]["score"] = np.mean([
            out["M5_both_time"]["both_rawFC"], out["M5_both_time"]["both_abs"],
            out["M5_both_time"]["time_abs"], out["M5_both_time"]["time_rawFC"]])

        w = self.cfg.weights
        out["per_split_rawFC"] = {k: self.m2_rawfc(P, self._sel(k))["score"]
                                  for k in (s1, s2, s3, stime)}
        out["TOTAL"] = float(sum(w[k] * out[k]["score"] for k in w))
        return out


def flatten(rep: dict) -> dict:
    flat = {"TOTAL": rep["TOTAL"]}
    for k, v in rep.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
    return flat

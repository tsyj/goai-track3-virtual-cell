"""Structured additive model for the yeast perturbation proteome.

Prediction is factorised the same way the scoring metric is:

    y_hat(sample) = B_hat(plate, strain)          <- absolute baseline level
                  + Delta_hat(compound, context)  <- perturbation effect

*B_hat* is a two-way (plate x strain) additive fit of **control wells**, on
protein-centred data, so a held-out strain still gets the correct plate/batch
level -- the single largest variance component (88% of total variance).  A
held-out strain's own term is unidentifiable and shrinks to the cross-strain
centre, exactly as it will for CRD at test time.

B_hat deliberately *smooths over* the matched control well rather than copying
it.  The official Delta metrics use the organiser's measured control on both
sides (Delta_pred = y_hat - y_ctrl, Delta_true = y_true - y_ctrl), so copying the
control well cancels a noise term the metric otherwise credits you for -- see
scripts/13_control_noise.py.

*Delta_hat* is fitted by shrunken backfitting over a hierarchy of design factors:

    global -> compound -> compound x {time, temperature, medium, source, strain}

For an unseen compound every compound-indexed term vanishes and only the shared
stress axis survives -- which is why module M3 (which subtracts the context mean)
needs chemistry, while M4 (which subtracts the drug mean) is solvable from
context modulation alone.

The whole task is transductive: test metadata is published, so all models are
fitted on the union of train+test *rows* while only train *labels* are visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
def interaction_codes(meta: pd.DataFrame, cols) -> tuple[np.ndarray, int]:
    """Integer codes for the interaction of ``cols`` over the given rows."""
    if not cols:
        return np.zeros(len(meta), np.int64), 1
    key = None
    for c in cols:
        cc = pd.factorize(meta[c].astype(str).to_numpy(), sort=True)[0]
        key = cc if key is None else key * (cc.max() + 1) + cc
    codes, uniq = pd.factorize(key, sort=True)
    return codes.astype(np.int64), len(uniq)


def _onehot(codes: np.ndarray, k: int, use: np.ndarray) -> sparse.csr_matrix:
    """(k, n) selector matrix restricted to rows where ``use`` is True."""
    rows = np.where(use)[0]
    return sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (codes[rows], rows)), shape=(k, len(codes)))


def shrunk_means(R: np.ndarray, S: sparse.csr_matrix, lam) -> np.ndarray:
    """Per-group mean of R (NaN = unobserved) shrunk toward 0 by n/(n+lam).

    ``lam`` may be a scalar or a per-protein vector (empirical Bayes).
    """
    M = np.isfinite(R)
    Rz = np.where(M, R, 0.0).astype(np.float32)
    return ((S @ Rz) / ((S @ M.astype(np.float32)) + lam)).astype(np.float32)


def eb_lambda(term: np.ndarray, counts: np.ndarray, sigma2: np.ndarray,
              floor: float = 0.25, cap: float = 400.0) -> np.ndarray:
    """Empirical-Bayes shrinkage strength per protein: lambda_p = sigma2_p / tau2_p.

    ``tau2_p`` is the between-level variance of the true term, recovered by
    subtracting the sampling variance already present in the fitted term.
    Proteins whose between-level spread is no larger than their own noise get a
    large lambda and are shrunk away; well-determined proteins keep their signal.
    """
    nbar = np.maximum(counts.mean(0), 1.0)
    var_hat = term.var(0)
    tau2 = np.maximum(var_hat - sigma2 / nbar, 1e-6)
    return np.clip(sigma2 / tau2, floor, cap).astype(np.float32)


def truncate_rank(T: np.ndarray, rank: int) -> np.ndarray:
    """Keep the leading ``rank`` singular directions of a (levels x proteins) term."""
    if rank <= 0 or rank >= min(T.shape):
        return T
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    return ((U[:, :rank] * S[:rank]) @ Vt[:rank]).astype(np.float32)


# ---------------------------------------------------------------------------
class ControlBaseline:
    """Two-way additive plate x strain model fitted on visible control wells."""

    def __init__(self, lam_plate: float = 1.0, lam_strain: float = 3.0, n_iter: int = 6,
                 controls_only: bool = True):
        self.lam_plate, self.lam_strain = lam_plate, lam_strain
        self.n_iter, self.controls_only = n_iter, controls_only

    def fit(self, meta: pd.DataFrame, Y_obs: np.ndarray):
        sel = np.isfinite(Y_obs).any(1)
        if self.controls_only:
            sel = sel & meta["is_control"].to_numpy()
        self.mu = np.nanmean(np.where(sel[:, None], Y_obs, np.nan), 0).astype(np.float32)

        self.pc, npl = interaction_codes(meta, ["Yeast_cell_plate"])
        self.sc, nst = interaction_codes(meta, ["Strains"])
        Sp, Ss = _onehot(self.pc, npl, sel), _onehot(self.sc, nst, sel)

        R = np.where(sel[:, None], Y_obs - self.mu, np.nan).astype(np.float32)
        self.plate = np.zeros((npl, R.shape[1]), np.float32)
        self.strain = np.zeros((nst, R.shape[1]), np.float32)
        for _ in range(self.n_iter):
            self.plate = shrunk_means(R - self.strain[self.sc], Sp, self.lam_plate)
            self.strain = shrunk_means(R - self.plate[self.pc], Ss, self.lam_strain)
        cnt = np.isfinite(R).astype(np.float32)
        self.strain_seen = ((Ss @ cnt) > 0).any(1)
        self.plate_seen = ((Sp @ cnt) > 0).any(1)
        return self

    def predict(self) -> np.ndarray:
        out = np.tile(self.mu, (len(self.pc), 1))
        ok = self.plate_seen[self.pc]
        out[ok] += self.plate[self.pc[ok]]
        ok = self.strain_seen[self.sc]
        out[ok] += self.strain[self.sc[ok]]
        return out


# ---------------------------------------------------------------------------
DEFAULT_FACTORS = [
    ("global", (), 0.0),
    ("compound", ("compound",), 8.0),
    ("cmpd_x_time", ("compound", "pert_time"), 12.0),
    ("cmpd_x_temp", ("compound", "Temperature"), 12.0),
    ("cmpd_x_medium", ("compound", "Medium"), 12.0),
    ("cmpd_x_source", ("compound", "data_source"), 12.0),
    ("cmpd_x_strain", ("compound", "Strains"), 18.0),
    ("strain", ("Strains",), 8.0),
    ("time", ("pert_time",), 8.0),
    ("temp", ("Temperature",), 8.0),
    ("medium", ("Medium",), 8.0),
]


class DeltaBackfit:
    """Shrunken backfitting of the perturbation effect over categorical factors.

    Transductive by construction: ``meta`` holds every row we will ever predict,
    ``use`` marks the rows whose labels may be learned from.
    """

    def __init__(self, factors=None, n_pass: int = 4):
        self.factors = factors or DEFAULT_FACTORS
        self.n_pass = n_pass

    def fit(self, meta: pd.DataFrame, D_obs: np.ndarray, use: np.ndarray):
        n, p = D_obs.shape
        self.codes, self.terms, self.S = {}, {}, {}
        for name, cols, _ in self.factors:
            c, k = interaction_codes(meta, cols)
            self.codes[name], self.terms[name] = c, np.zeros((k, p), np.float32)
            self.S[name] = _onehot(c, k, use)

        R = np.where(use[:, None], D_obs, np.nan).astype(np.float32)
        total = np.zeros((n, p), np.float32)
        for _ in range(self.n_pass):
            for name, _, lam in self.factors:
                cur = self.terms[name][self.codes[name]]
                new = shrunk_means(R - (total - cur), self.S[name], lam)
                total += new[self.codes[name]] - cur
                self.terms[name] = new
        self._total = total
        return self

    def predict(self) -> np.ndarray:
        return self._total

    def term_norms(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"term": k, "n_levels": v.shape[0], "rms": float(np.sqrt((v ** 2).mean()))}
            for k, v in self.terms.items()]).sort_values("rms", ascending=False)


# ---------------------------------------------------------------------------
# Unified model: one backfit over batch structure *and* perturbation structure.
#
# Fitting the plate level on control wells alone wastes 87% of the labels -- the
# perturbation effect (rms 0.15 log2) is small next to the plate effect, so every
# well on a plate informs that plate's level.  Terms are ordered coarse -> fine;
# terms indexed by a held-out entity simply evaluate to zero for it.

BATCH_FACTORS = [
    # Two coarse parents above plate.  The ladder used to jump straight from the
    # global mean to a 144-level plate term; inserting data_source (4 levels) and
    # instrument (7) is worth +0.00171 +- 0.00030 on the adopted booster, 3/3 folds
    # (scripts/42_confirm_instrument.py).  Both are perfectly nested in plate, which
    # is why they were never tried -- but nesting removes the *information*, not the
    # partial pooling.  They must be fitted BEFORE plate: moved after it the gain
    # collapses from +0.0056 to +0.0013 (scripts/41_instrument_level.py), which is
    # the signature of hierarchical shrinkage rather than new signal.  lambda is
    # immaterial here (1 / 3 / 8 / 20 all within noise) -- these terms are estimated
    # from thousands of wells each.
    #
    # NOTE the headline numbers in 39/41 (+0.0056 to +0.0071) are inflated: three of
    # the six inner mirrors put ~32% of their evaluation rows on plates carrying no
    # training label at all, a regime that does not exist in the official split
    # (0 of 4,454 test rows).  Judge anything that acts through the plate term on the
    # orphan-free folds only -- scripts/analyze_paired.py reports both columns.
    ("source", ("data_source",), 3.0),
    ("instrument", ("instrument",), 3.0),
    ("plate", ("Yeast_cell_plate",), 1.0),
    ("plate_x_strain", ("Yeast_cell_plate", "Strains"), 6.0),
    ("strain", ("Strains",), 3.0),
    ("strain_x_medium", ("Strains", "Medium"), 6.0),
    ("strain_x_temp", ("Strains", "Temperature"), 6.0),
    ("strain_x_time", ("Strains", "pert_time"), 8.0),
    ("strain_x_source", ("Strains", "data_source"), 8.0),
]

PERT_FACTORS = [
    ("compound", ("compound",), 8.0),
    ("cmpd_x_time", ("compound", "pert_time"), 12.0),
    ("cmpd_x_temp", ("compound", "Temperature"), 12.0),
    ("cmpd_x_medium", ("compound", "Medium"), 12.0),
    ("cmpd_x_source", ("compound", "data_source"), 12.0),
    ("cmpd_x_strain", ("compound", "Strains"), 18.0),
]


class UnifiedBackfit:
    """Y = mu[p] + offset[i] + sum_f term_f[level_f(i), p], shrunken backfitting.

    ``offset`` is a per-sample scalar (protein-loading nuisance).  It is fitted
    only where labels are visible and is zero for held-out rows -- which is
    correct, because it is unpredictable there and, being constant across
    proteins, it cancels out of every correlation-based module anyway.
    """

    def __init__(self, batch_factors=None, pert_factors=None, n_pass: int = 6,
                 fit_offset: bool = True, pert_scale: float = 1.0,
                 lowrank: dict | None = None, eb: bool = False):
        self.factors = list(batch_factors if batch_factors is not None else BATCH_FACTORS)
        self.pert = list(pert_factors if pert_factors is not None else PERT_FACTORS)
        self.n_pass, self.fit_offset, self.pert_scale = n_pass, fit_offset, pert_scale
        self.eb = eb
        # term name -> retained rank.  The compound response space is strongly
        # low-dimensional (5 PCs ~ 80% of variance, and a rank-5 reconstruction
        # already correlates 0.70 with a held-out half of the same compound's
        # samples), so truncating discards mostly estimation noise.
        self.lowrank = lowrank or {}

    def fit(self, meta: pd.DataFrame, Y_obs: np.ndarray, use: np.ndarray):
        """Backfit with the observation mask factored out of the inner loop.

        The NaN pattern of the target never changes, so ``S @ Rz`` and the
        per-level observation counts are constants.  Because a term is constant
        within its own level, ``S @ (M * cur)`` collapses to ``counts * term``.
        Each factor update then costs one sparse mat-mult plus one elementwise
        multiply over the data matrix, instead of six full temporaries.
        """
        n, p = Y_obs.shape
        allf = self.factors + self.pert
        self.pert_names = {name for name, _, _ in self.pert}
        self.mu = np.nanmean(np.where(use[:, None], Y_obs, np.nan), 0).astype(np.float32)
        # 186 of the 5,243 proteins are never quantified in any visible sample, so
        # they have no mean.  Missingness here is not at random -- it tracks low
        # abundance (corr -0.66 with mean log2) -- so the honest fill is the low
        # end of the observed distribution, not zero and not the grand mean.
        self.mu_fallback = float(np.nanpercentile(
            np.where(use[:, None], Y_obs, np.nan), 5))
        self.mu_missing = ~np.isfinite(self.mu)
        self.mu = np.where(self.mu_missing, self.mu_fallback, self.mu).astype(np.float32)

        obs = np.isfinite(Y_obs) & use[:, None]
        M = obs.astype(np.float32)
        Rz = np.where(obs, Y_obs - self.mu, 0.0).astype(np.float32)

        self.codes, self.terms, self.S = {}, {}, {}
        SR, CNT = {}, {}
        for name, cols, _ in allf:
            c, k = interaction_codes(meta, cols)
            self.codes[name] = c
            self.terms[name] = np.zeros((k, p), np.float32)
            self.S[name] = _onehot(c, k, use)
            SR[name] = self.S[name] @ Rz
            CNT[name] = self.S[name] @ M

        self.offset = np.zeros(n, np.float32)
        total = np.zeros((n, p), np.float32)
        Mtotal = np.zeros((n, p), np.float32)
        lams = {name: lam for name, _, lam in allf}
        rounds = [self.n_pass] + ([self.n_pass] if self.eb else [])
        for rnd, npass in enumerate(rounds):
            if rnd == 1:                       # re-shrink using empirical Bayes
                nobs = np.maximum(M.sum(0), 1.0)
                sigma2 = ((Rz - Mtotal) ** 2).sum(0) / nobs
                for name, _, _ in allf:
                    lams[name] = eb_lambda(self.terms[name], CNT[name], sigma2)
                self.sigma2 = sigma2
            for _ in range(npass):
                for name, _, _ in allf:
                    codes, T, lam = self.codes[name], self.terms[name], lams[name]
                    new = ((SR[name] - self.S[name] @ Mtotal + CNT[name] * T)
                           / (CNT[name] + lam)).astype(np.float32)
                    d = new[codes] - T[codes]
                    total += d
                    Mtotal += M * d
                    self.terms[name] = new
                if self.fit_offset:
                    cnt = M.sum(1)
                    newo = (((Rz - Mtotal).sum(1)) / np.maximum(cnt, 1)).astype(np.float32)
                    newo[cnt == 0] = 0.0
                    total += (newo - self.offset)[:, None]
                    Mtotal += M * (newo - self.offset)[:, None]
                    self.offset = newo
        self.lams = lams
        for name, rank in self.lowrank.items():
            if name in self.terms:
                old = self.terms[name]
                self.terms[name] = truncate_rank(old, rank)
                total += self.terms[name][self.codes[name]] - old[self.codes[name]]
        self._total = total
        return self

    def predict(self, pert_scale: float | None = None) -> np.ndarray:
        s = self.pert_scale if pert_scale is None else pert_scale
        out = self.mu + self._batch_part()
        if s:
            out = out + s * self.pert_part()
        return np.where(np.isfinite(out), out, self.mu_fallback).astype(np.float32)

    def _batch_part(self) -> np.ndarray:
        n = len(self.offset)
        out = np.zeros((n, self.mu.shape[0]), np.float32)
        for name, _, _ in self.factors:
            out += self.terms[name][self.codes[name]]
        return out

    def pert_part(self) -> np.ndarray:
        n = len(self.offset)
        out = np.zeros((n, self.mu.shape[0]), np.float32)
        for name in self.pert_names:
            out += self.terms[name][self.codes[name]]
        return out

    def term_norms(self) -> pd.DataFrame:
        rows = [{"term": k, "n_levels": v.shape[0], "kind": "pert" if k in
                 self.pert_names else "batch", "rms": float(np.sqrt((v ** 2).mean()))}
                for k, v in self.terms.items()]
        rows.append({"term": "offset", "n_levels": len(self.offset), "kind": "nuisance",
                     "rms": float(np.sqrt((self.offset ** 2).mean()))})
        return pd.DataFrame(rows).sort_values("rms", ascending=False)


# ---------------------------------------------------------------------------
class ResidualBooster:
    """Gradient boosting on what the additive model leaves behind.

    The additive model handles the batch structure with terms trees cannot match
    (144 plates x 5,243 proteins), but it can only express effects that are sums
    of one-factor and two-factor tables.  Boosting on its residual picks up the
    higher-order interactions -- and measurably does: on the inner mirror it
    lifts the total by ~0.013, most of it in the two residual modules M3 and M4,
    which the additive model alone cannot move at all for a held-out compound.

    5,243 outputs are compressed to the leading principal components of the
    residual first; the proteome residual is low-rank enough that ~96 components
    carry what is learnable, and one booster per component is cheap.
    """

    CAT = ["Strains", "compound", "Medium", "data_source", "instrument",
           "Yeast_cell_plate", "well_row"]
    NUM = ["Temperature", "pert_time", "well_col"]

    def __init__(self, n_comp: int = 96, n_estimators: int = 800,
                 learning_rate: float = 0.03, num_leaves: int = 31,
                 scale: float = 1.0, n_jobs: int = 8, seed: int = 0,
                 seeds=None):
        self.n_comp, self.n_estimators = n_comp, n_estimators
        self.learning_rate, self.num_leaves = learning_rate, num_leaves
        self.scale, self.n_jobs, self.seed = scale, n_jobs, seed
        # LightGBM here is stochastic (subsample .8, colsample .9), so one fit is
        # one draw.  Averaging several draws removes that variance: +0.0008 +- 0.0001
        # paired over six inner folds, 6/6 (scripts/33_booster_knobs.py).  The PCA
        # basis is deterministic, so averaging the component scores is the same as
        # averaging the reconstructed proteomes.
        self.seeds = [int(s) for s in seeds] if seeds is not None else [int(seed)]

    @classmethod
    def _raw_column(cls, meta: pd.DataFrame, c: str) -> pd.Series:
        if c == "well_row":
            return meta["protein_well"].str.extract(r"^([A-H])")[0]
        return meta[c]

    def featurise(self, meta: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode metadata.  Category codes are frozen at fit time.

        Recomputing ``pd.Categorical(...).codes`` on a different frame would
        silently renumber every category -- the same string could map to a
        different integer than the trees were trained on.  So the level order is
        stored on fit and reused; a value unseen at fit time encodes to -1.
        """
        if fit:
            self._levels = {c: pd.Index(sorted(set(
                self._raw_column(meta, c).astype(str)))) for c in self.CAT}
        X = pd.DataFrame(index=meta.index)
        for c in self.CAT:
            X[c] = self._levels[c].get_indexer(
                self._raw_column(meta, c).astype(str))
        X["Temperature"] = meta["Temperature"].astype(float)
        X["pert_time"] = np.log2(meta["pert_time"].astype(float))
        X["well_col"] = meta["protein_well"].str.extract(r"(\d+)$")[0].astype(float)
        return X

    def fit(self, meta: pd.DataFrame, Y_obs: np.ndarray, use: np.ndarray,
            base: np.ndarray):
        import lightgbm as lgb
        R = np.where(np.isfinite(Y_obs) & use[:, None], Y_obs - base, np.nan)
        Rv = np.nan_to_num(R[use]).astype(np.float32)
        U, S, Vt = np.linalg.svd(Rv, full_matrices=False)
        k = min(self.n_comp, Vt.shape[0])
        self.V = Vt[:k]
        self.explained = float((S[:k] ** 2).sum() / (S ** 2).sum())
        Z = Rv @ self.V.T

        X = self.featurise(meta, fit=True)
        self.model_sets = []
        for s in self.seeds:
            models = []
            for j in range(k):
                g = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves, min_child_samples=30, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                    random_state=s + j, verbose=-1, n_jobs=self.n_jobs)
                g.fit(X[use], Z[:, j], categorical_feature=self.CAT)
                models.append(g)
            self.model_sets.append(models)
        self.models = self.model_sets[0]      # kept for callers that report len(models)
        self._X = X
        return self

    def predict(self, meta: pd.DataFrame | None = None) -> np.ndarray:
        X = self._X if meta is None else self.featurise(meta, fit=False)
        Z = np.mean([np.column_stack([g.predict(X) for g in ms])
                     for ms in self.model_sets], axis=0).astype(np.float32)
        return self.scale * (Z @ self.V)

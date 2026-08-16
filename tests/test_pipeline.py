"""Tests for the things that would silently corrupt a submission.

Run: /home/xinyuan/anaconda3/envs/numpy1/bin/python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.design import match_controls                                   # noqa: E402
from vcell.io import load_metadata, load_proteome                         # noqa: E402
from vcell.metrics import _average_precision, _pcc_rows, _r2_rows         # noqa: E402
from vcell.models import ResidualBooster, UnifiedBackfit                  # noqa: E402


# --------------------------------------------------------------- metric maths
def test_pcc_matches_scipy():
    from scipy.stats import pearsonr
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 200))
    B = 0.4 * A + rng.normal(size=(20, 200))
    A[rng.random(A.shape) < 0.2] = np.nan
    B[rng.random(B.shape) < 0.2] = np.nan
    ours = _pcc_rows(A, B)
    for i in range(len(A)):
        m = np.isfinite(A[i]) & np.isfinite(B[i])
        assert ours[i] == pytest.approx(pearsonr(A[i][m], B[i][m])[0], abs=1e-12)


def test_r2_matches_sklearn():
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(1)
    P = rng.normal(size=(15, 150))
    T = P + rng.normal(scale=0.5, size=(15, 150))
    P[rng.random(P.shape) < 0.2] = np.nan
    ours = _r2_rows(P, T)
    for i in range(len(P)):
        m = np.isfinite(P[i]) & np.isfinite(T[i])
        assert ours[i] == pytest.approx(r2_score(T[i][m], P[i][m]), abs=1e-12)


def test_average_precision_matches_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(2)
    s, y = rng.random(300), (rng.random(300) < 0.25).astype(float)
    assert _average_precision(s, y) == pytest.approx(average_precision_score(y, s))


def test_degenerate_rows_are_nan_not_zero():
    """A constant prediction has no correlation -- it must not score as 0.0."""
    z = np.zeros((2, 40))
    r = rng_pair = _pcc_rows(z, np.random.default_rng(3).normal(size=(2, 40)))
    assert np.all(np.isnan(r)), rng_pair


# ------------------------------------------------------------- data integrity
def test_test_proteome_is_unreachable():
    """The leaked test labels must not be loadable through the normal path."""
    with pytest.raises(RuntimeError):
        load_proteome("test")


def test_every_treated_sample_has_a_matched_control():
    meta = load_metadata("both").reset_index(drop=True)
    rows = match_controls(meta, strategy="both")
    treated = ~meta["is_control"] & ~meta["is_qc"]
    assert all(len(rows[i]) > 0 for i in np.where(treated)[0])


def test_a_control_is_not_its_own_control():
    meta = load_metadata("both").reset_index(drop=True)
    rows = match_controls(meta, strategy="both")
    for i in np.where(meta["is_control"])[0]:
        assert i not in rows[i]


def test_compound_identity_is_not_pert_id():
    """pert_id is reused across data sources; the compound name is the identity."""
    meta = load_metadata("both")
    per_id = meta.groupby("pert_id")["compound"].nunique()
    assert (per_id > 1).any(), "expected pert_id to be ambiguous across sources"


# ----------------------------------------------------------------- model API
@pytest.fixture(scope="module")
def small_fit():
    P = load_proteome("train_val")
    meta = P.meta.iloc[:1200].reset_index(drop=True)
    Y = P.X[:1200]
    use = (meta["split_final"] == "train").to_numpy()
    um = UnifiedBackfit(n_pass=2).fit(meta, np.where(use[:, None], Y, np.nan), use)
    return meta, Y, use, um


def test_fit_is_deterministic(small_fit):
    meta, Y, use, um = small_fit
    again = UnifiedBackfit(n_pass=2).fit(meta, np.where(use[:, None], Y, np.nan), use)
    assert np.allclose(um.predict(), again.predict(), equal_nan=True)


def test_prediction_has_no_nan(small_fit):
    _, _, _, um = small_fit
    assert np.isfinite(um.predict()).all()


def test_held_out_entities_get_no_compound_term(small_fit):
    """A compound with no visible label must contribute exactly zero.

    'Visible' means any labelled row, including control wells: Water and DMSO are
    compounds too, and the model legitimately fits a term for them describing how
    a control well sits relative to its plate average.
    """
    meta, _, use, um = small_fit
    seen = set(meta.loc[use, "compound"])
    unseen = np.where(~meta["compound"].isin(seen).to_numpy())[0]
    assert len(unseen) > 0, "fixture must contain at least one held-out compound"
    assert np.allclose(um.pert_part()[unseen], 0.0)


def test_booster_category_encoding_is_frozen_at_fit(small_fit):
    """Regression test.

    featurise() used to rebuild pd.Categorical codes from whatever frame it was
    handed, so predicting on a re-ordered frame silently renumbered every
    category.  Predicting on a permuted frame must now give the permuted rows of
    the original prediction, not different values.
    """
    meta, Y, use, um = small_fit
    base = um.predict()
    rb = ResidualBooster(n_comp=4, n_estimators=25, n_jobs=2)
    rb.fit(meta, np.where(use[:, None], Y, np.nan), use, base)
    straight = rb.predict()

    perm = np.random.default_rng(0).permutation(len(meta))
    shuffled = rb.predict(meta.iloc[perm].reset_index(drop=True))
    assert np.allclose(shuffled, straight[perm], atol=1e-5)

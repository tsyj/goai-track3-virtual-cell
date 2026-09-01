"""Data loading + caching for the GOAI virtual-cell (yeast perturbation proteome) task.

Design rules enforced here
--------------------------
* The public download shipped the test-set proteome *with ground-truth labels*.
  That file has been moved to ``data/quarantine/`` and is NEVER read by this module.
  ``load_proteome('test')`` raises.  A separate, clearly-marked audit script is the
  only place allowed to touch it.
* Everything is cached as float32 ``.npy`` + a parquet/csv metadata frame so the
  8958 x 5243 matrix loads in <1 s.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

def _find_data_root():
    """定位包含 data/input 的项目根。

    环境变量 VCELL_DATA_ROOT 优先；否则从本文件向上逐级寻找 data/input。这样无论
    vcell/ 放在仓库根下还是 src/ 下，也无论评审把官方数据挂到哪里，都能找到。
    """
    env = os.environ.get("VCELL_DATA_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        here = os.path.dirname(here)
        if not here or here == os.sep:
            break
        if os.path.isdir(os.path.join(here, "data", "input")):
            return here
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _find_data_root()
INPUT = os.path.join(ROOT, "data", "input")
CACHE = os.path.join(ROOT, "data", "cache")
QUARANTINE = os.path.join(ROOT, "data", "quarantine")

META_TRAIN = os.path.join(INPUT, "WAYB_WAYC_metadata_train_val(1).csv")
META_TEST = os.path.join(INPUT, "WAYB_WAYC_metadata_test(1).csv")
PROT_TRAIN = os.path.join(INPUT, "WAYB_WAYC_proteome_raw_train_val.csv")

# Columns that define the "measurement context" a control must be matched on.
CONTROL_KEYS = [
    "data_source",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "instrument",
    "Yeast_cell_plate",
]
CONTROL_COMPOUNDS = ("DMSO", "Water")
QC_COMPOUND = "Quality Control"


def _sha256(path: str, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(nbytes):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(which: str = "both") -> pd.DataFrame:
    """Return metadata with a ``SET`` column in {train_val, test}."""
    frames = []
    if which in ("train_val", "both"):
        a = pd.read_csv(META_TRAIN)
        a["SET"] = "train_val"
        frames.append(a)
    if which in ("test", "both"):
        b = pd.read_csv(META_TEST)
        b["SET"] = "test"
        frames.append(b)
    m = pd.concat(frames, ignore_index=True)
    # canonical compound identity: pert_id is only unique *within* a data_source
    m["compound"] = m["perturbation_no_concentration"].astype(str)
    m["is_control"] = m["compound"].isin(CONTROL_COMPOUNDS)
    m["is_qc"] = m["compound"] == QC_COMPOUND
    m["ctx_key"] = m[CONTROL_KEYS].astype(str).agg("|".join, axis=1)
    # context excluding strain -- used for plate/run level effects
    m["batch_key"] = m[["data_source", "Medium", "Temperature", "pert_time",
                        "instrument", "Yeast_cell_plate"]].astype(str).agg("|".join, axis=1)
    return m


def build_cache(force: bool = False) -> None:
    """log2-transform the train_val proteome and cache as .npy."""
    xpath = os.path.join(CACHE, "train_val_log2.npy")
    if os.path.exists(xpath) and not force:
        return
    os.makedirs(CACHE, exist_ok=True)
    df = pd.read_csv(PROT_TRAIN)
    ids = df["sample_ID"].astype(str).to_numpy()
    proteins = np.array(df.columns[1:], dtype=object)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    del df
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.log2(X, out=X, where=np.isfinite(X) & (X > 0))
    np.save(xpath, X)
    np.save(os.path.join(CACHE, "train_val_sample_ids.npy"), ids)
    np.save(os.path.join(CACHE, "proteins.npy"), proteins)
    with open(os.path.join(CACHE, "PROVENANCE.txt"), "w") as fh:
        fh.write(f"source={PROT_TRAIN}\nsha256={_sha256(PROT_TRAIN)}\n"
                 f"shape={X.shape}\ntransform=log2(raw)\n")


@dataclass
class Proteome:
    X: np.ndarray            # (n_samples, n_proteins) log2 abundance, NaN = missing
    sample_ids: np.ndarray   # (n_samples,)
    proteins: np.ndarray     # (n_proteins,)
    meta: pd.DataFrame       # aligned row-for-row with X

    def __len__(self) -> int:
        return self.X.shape[0]


def load_combined() -> Proteome:
    """train_val rows with labels + test rows with all-NaN labels.

    The task is transductive -- test metadata is published -- so the model is
    fitted over the union of rows while only train_val labels are visible.  The
    test proteome file is never opened; its rows are NaN by construction.
    """
    tv = load_proteome("train_val")
    mte = load_metadata("test")
    meta = pd.concat([tv.meta, mte], ignore_index=True)
    X = np.full((len(meta), tv.X.shape[1]), np.nan, dtype=np.float32)
    X[: len(tv.meta)] = tv.X
    ids = np.concatenate([tv.sample_ids, mte["sample_ID"].astype(str).to_numpy()])
    return Proteome(X=X, sample_ids=ids, proteins=tv.proteins, meta=meta)


def load_proteome(which: str = "train_val") -> Proteome:
    if which != "train_val":
        raise RuntimeError(
            "Only 'train_val' may be loaded. The test proteome shipped with leaked "
            "ground-truth labels and lives in data/quarantine/; it must not enter "
            "training or model selection."
        )
    build_cache()
    X = np.load(os.path.join(CACHE, "train_val_log2.npy"))
    ids = np.load(os.path.join(CACHE, "train_val_sample_ids.npy"), allow_pickle=True)
    proteins = np.load(os.path.join(CACHE, "proteins.npy"), allow_pickle=True)
    meta = load_metadata("train_val").set_index("sample_ID").loc[ids].reset_index()
    assert (meta["sample_ID"].to_numpy() == ids).all()
    return Proteome(X=X, sample_ids=ids, proteins=proteins, meta=meta)

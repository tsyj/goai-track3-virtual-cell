"""Chemical similarity between perturbagens -- the only route to an unseen compound.

For a compound with no labelled sample, every compound-indexed term of the
additive model is zero by construction.  The effect vector has to be borrowed
from chemically related compounds that *were* measured:

    alpha_hat(c*) = sum_c w(c*, c) alpha(c),      w from a similarity kernel

Two similarity sources, both public knowledge (open-knowledge board):
  * structural  -- Tanimoto over Morgan fingerprints of PubChem SMILES
  * mechanistic -- a curated mode-of-action class per compound

The held-out compounds were not chosen at random: Tamoxifen is held out while
4-Hydroxytamoxifen (its active metabolite) is in training, Fluconazole is held
out while the azole Clotrimazole is in training, three aminoglycosides are held
out while cycloheximide and anisomycin are in training.  Mechanistic borrowing is
therefore expected to carry real signal -- scripts/05_chem_transfer.py measures
how much, by leave-one-compound-out over the *training* compounds only.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEM = os.path.join(ROOT, "data", "chem")

# Curated mode-of-action classes.  Sources: standard pharmacology / yeast
# chemogenomics literature; one line per compound so the table can be audited.
MOA = {
    "Water": "control", "DMSO": "control", "Quality Control": "control",
    # translation
    "CHX": "translation_cytosolic", "Anisomycin": "translation_cytosolic",
    "G418": "aminoglycoside", "Hygromycin B": "aminoglycoside",
    "Neomycin B": "aminoglycoside",
    "Doxycycline hyclate": "translation_mitochondrial",
    # DNA damage / replication
    "MMS": "dna_damage_alkylating", "Cisplatin": "dna_damage_crosslink",
    "(S)-(+)-Camptothecin": "dna_damage_topoisomerase",
    "Hydroxyurea": "replication_stress", "Hoechst 33258": "dna_binder",
    "Pentamidine isethionate": "dna_binder",
    # ergosterol / membrane
    "Fluconazole": "ergosterol_azole", "Clotrimazole": "ergosterol_azole",
    "Amphotericin B": "ergosterol_polyene", "Nystatin dihydrate": "ergosterol_polyene",
    "SDS": "membrane_detergent", "Abietic acid": "membrane",
    "(1R, 2S, 5R) - (-) - Menthol": "membrane",
    "Dyclonine hydrochloride": "membrane",
    "Amiodarone hydrochloride": "cationic_amphiphile",
    "Desipramine hydrochloride": "cationic_amphiphile",
    "Trifluoperazine dihydrochloride": "cationic_amphiphile",
    "Haloperidol": "cationic_amphiphile",
    # oxidative / redox
    "H2O2": "oxidative", "Plumbagin": "oxidative", "Emodin": "oxidative",
    "Artemisinin": "oxidative", "Parthenolide": "oxidative_thiol",
    # mitochondria / bioenergetics
    "FCCP": "mito_uncoupler", "Oligomycin": "mito_atp_synthase",
    "Nigericin": "ionophore", "Valinomycin": "ionophore",
    # signalling
    "Rapamycin": "tor", "LY 294002 hydrochloride": "pi3k_tor",
    "Wortmannin": "pi3k_tor", "Staurosporine": "kinase_broad",
    "Harmine hydrochloride": "kinase_broad", "U-73122": "lipid_signalling",
    "Cyclopiazonic acid": "calcium_atpase",
    # SERM / tamoxifen family
    "Tamoxifen": "serm", "4-Hydroxytamoxifen": "serm",
    "Raloxifene hydrochloride": "serm", "Clomiphene citrate": "serm",
    # proteostasis / secretion
    "Geldanamycin": "hsp90", "Tunicamycin": "er_stress_upr",
    "Brefeldin A": "secretion_er_golgi", "Trichostatin A": "hdac",
    "Nocodazole": "microtubule",
    # metabolic / ionic
    "Sulfometuron methyl": "aa_biosynthesis", "EDTA": "metal_chelation",
    "1-10 Phenanthroline monohydrate": "metal_chelation",
    "NaCl": "osmotic_ionic", "Sorbitol": "osmotic",
}


def load_pubchem() -> pd.DataFrame:
    with open(os.path.join(CHEM, "pubchem.json")) as fh:
        df = pd.DataFrame(json.load(fh))
    df["moa"] = df["name"].map(MOA).fillna("unknown")
    return df


def fingerprints(df: pd.DataFrame, radius: int = 2, nbits: int = 2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps, ok = [], []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            fps.append(np.zeros(nbits, np.uint8)); ok.append(False)
        else:
            fps.append(np.frombuffer(gen.GetFingerprintAsNumPy(mol).tobytes(),
                                     dtype=np.uint8)[:nbits])
            ok.append(True)
    return np.asarray(fps, np.float32), np.asarray(ok)


def tanimoto(F: np.ndarray) -> np.ndarray:
    inter = F @ F.T
    n = F.sum(1)
    union = n[:, None] + n[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        T = inter / union
    return np.nan_to_num(T)


def similarity_matrix(df: pd.DataFrame, w_struct: float = 0.5,
                      w_moa: float = 0.5) -> np.ndarray:
    """Combined structural + mechanistic similarity, diagonal included."""
    F, _ = fingerprints(df)
    T = tanimoto(F)
    moa = df["moa"].to_numpy()
    same = (moa[:, None] == moa[None, :]).astype(np.float32)
    same[np.array([m == "unknown" for m in moa])[:, None] |
         np.array([m == "unknown" for m in moa])[None, :]] = 0.0
    return w_struct * T + w_moa * same


def transfer_weights(S: np.ndarray, targets: np.ndarray, donors: np.ndarray,
                     topk: int = 5, temp: float = 0.15,
                     min_sim: float = 0.05) -> np.ndarray:
    """(n_targets, n_donors) softmax-over-top-k weights, rows sum to 1 (or 0)."""
    W = np.zeros((len(targets), len(donors)), np.float32)
    for i, t in enumerate(targets):
        s = S[t, donors].astype(np.float64).copy()
        s[donors == t] = -np.inf
        s[s < min_sim] = -np.inf
        if not np.isfinite(s).any():
            continue
        k = min(topk, int(np.isfinite(s).sum()))
        keep = np.argpartition(-s, k - 1)[:k]
        w = np.zeros_like(s)
        e = np.exp((s[keep] - s[keep].max()) / temp)
        w[keep] = e / e.sum()
        W[i] = w
    return W

"""Resolve the 57 perturbagen names to PubChem CIDs / SMILES (open-knowledge board).

Everything fetched here is public chemistry, logged with its source URL so the
external-resource disclosure required by the handbook can be generated verbatim.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.io import load_metadata  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "chem")
os.makedirs(OUT, exist_ok=True)
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# names PubChem cannot resolve verbatim -> the query it does understand
ALIAS = {
    "CHX": "cycloheximide",
    "MMS": "methyl methanesulfonate",
    "SDS": "sodium dodecyl sulfate",
    "H2O2": "hydrogen peroxide",
    "G418": "geneticin",
    "EDTA": "edetic acid",
    "FCCP": "carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone",
    "NaCl": "sodium chloride",
    "DMSO": "dimethyl sulfoxide",
    "Water": "water",
    "(1R, 2S, 5R) - (-) - Menthol": "levomenthol",
    "(S)-(+)-Camptothecin": "camptothecin",
    "1-10 Phenanthroline monohydrate": "1,10-phenanthroline monohydrate",
    "LY 294002 hydrochloride": "LY-294002 hydrochloride",
    "U-73122": "U-73122",
    "Hoechst 33258": "Hoechst 33258",
    "Oligomycin": "oligomycin A",
    "Tunicamycin": "tunicamycin A",
    "Quality Control": None,          # not a compound
}


def get(url: str, tries: int = 3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:                                   # noqa: BLE001
            if k == tries - 1:
                return {"__error__": str(e)}
            time.sleep(1.5 * (k + 1))
    return None


def resolve(name: str) -> dict:
    q = ALIAS.get(name, name)
    if q is None:
        return {"name": name, "cid": None, "note": "not a chemical"}
    url = (f"{BASE}/compound/name/{urllib.parse.quote(q)}/property/"
           "SMILES,MolecularFormula,MolecularWeight,IUPACName/JSON")
    js = get(url)
    if not js or "PropertyTable" not in js:
        return {"name": name, "query": q, "cid": None,
                "error": (js or {}).get("__error__", "no match")}
    p = js["PropertyTable"]["Properties"][0]
    return {"name": name, "query": q, "cid": p.get("CID"),
            "smiles": p.get("SMILES") or p.get("ConnectivitySMILES"), "formula": p.get("MolecularFormula"),
            "mw": p.get("MolecularWeight"), "iupac": p.get("IUPACName"),
            "source": url}


if __name__ == "__main__":
    for v in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        os.environ.setdefault(v, "http://127.0.0.1:8118")
    names = sorted(load_metadata("both")["compound"].unique())
    print(f"resolving {len(names)} perturbagen names against PubChem ...")
    recs, bad = [], []
    for i, nm in enumerate(names, 1):
        r = resolve(nm)
        recs.append(r)
        flag = "ok " if r.get("smiles") else ("--" if r.get("cid") is None and
                                              r.get("note") else "FAIL")
        if flag == "FAIL":
            bad.append(nm)
        print(f"  [{i:2d}/{len(names)}] {flag} {nm[:44]:46s} "
              f"CID={str(r.get('cid')):>10s}")
        time.sleep(0.25)
    with open(os.path.join(OUT, "pubchem.json"), "w") as fh:
        json.dump(recs, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}/pubchem.json   unresolved={bad}")

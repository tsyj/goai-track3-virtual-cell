"""Score singles and ensembles from the saved member pool (scripts/55_member_pool.py).

Everything is paired per fold and reported as mean delta / sem / folds_up over the
six orphan-free folds, exactly like analyze_paired.py's free_delta column.

    python scripts/56_pool_eval.py --tag cheap \
        --base A,B,C,D,E --ref A [--greedy 4] [--combos "A,B,C;A,D,E"]

Reports
  1. singles vs --ref                       (is any candidate better on its own?)
  2. base ensemble vs --ref, base minus each member (does each earn its place?)
  3. base + each non-member                 (who is worth adding?)
  4. greedy forward selection from base     (--greedy K steps; selection noise on
                                             six folds is real -- read the sems)
  5. any explicit --combos

Jiao Xinyuan 2026-08-16 (evening session)
"""
import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import INNER, build_fold, evaluate, make_inner_splits  # noqa: E402

OUT = os.path.join(ROOT, "results")
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]

_FOLD_CACHE = {}


FOLD_PKL = os.path.join(OUT, "folds")


def _fold(seed, strain):
    """Fold objects take ~30 s to build; cache them as pickles (results/folds/)."""
    key = (seed, strain)
    if key not in _FOLD_CACHE:
        import pickle
        os.makedirs(FOLD_PKL, exist_ok=True)
        pk = os.path.join(FOLD_PKL, f"{seed}_{strain}.pkl")
        if os.path.exists(pk):
            with open(pk, "rb") as fh:
                _FOLD_CACHE[key] = pickle.load(fh)
        else:
            base_meta = build_fold().meta
            fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
            tmp = f"{pk}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(fo, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, pk)
            _FOLD_CACHE[key] = fo
    return _FOLD_CACHE[key]


def _pred(pool_dir, seed, strain, member):
    return np.load(os.path.join(pool_dir, f"{seed}_{strain}__{member}.npy"), mmap_mode="r")


def score_task(arg):
    pool_dir, seed, strain, combo = arg
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    fo = _fold(seed, strain)
    acc = None
    for m in combo:
        p = np.asarray(_pred(pool_dir, seed, strain, m), dtype=np.float64)
        acc = p if acc is None else acc + p
    P = (acc / len(combo)).astype(np.float32)
    rep = evaluate(fo, P, INNER)
    return {"seed": seed, "strain": strain, "combo": "+".join(combo), "TOTAL": rep["TOTAL"],
            "M1": rep["M1_absolute"]["score"], "M2": rep["M2_rawFC"]["score"],
            "M3": rep["M3_ctx_resid"]["score"], "M4": rep["M4_drug_resid"]["score"],
            "M5": rep["M5_both_time"]["score"], "M6": rep["M6_high_effect"]["score"]}


class Evaluator:
    def __init__(self, pool_dir, folds, n_workers):
        self.pool_dir, self.folds = pool_dir, folds
        self.pool = Pool(n_workers)
        self.cache = {}       # combo tuple -> Series indexed by (seed, strain)

    def score(self, combos):
        combos = [tuple(c) for c in combos]
        todo = [c for c in dict.fromkeys(combos) if c not in self.cache]
        tasks = [(self.pool_dir, s, st, c) for c in todo for (s, st) in self.folds]
        if tasks:
            rows = self.pool.map(score_task, tasks, chunksize=1)
            df = pd.DataFrame(rows)
            for c in todo:
                sub = df[df.combo == "+".join(c)].set_index(["seed", "strain"])
                self.cache[c] = sub
        return {c: self.cache[c] for c in combos}

    def close(self):
        self.pool.close()
        self.pool.join()


def paired_table(ev, combos, ref, label_of=lambda c: "+".join(c)):
    res = ev.score(list(combos) + [ref])
    r = res[tuple(ref)]["TOTAL"]
    rows = []
    for c in combos:
        c = tuple(c)
        d = res[c]["TOTAL"].reindex(r.index) - r
        row = {"config": label_of(c), "mean": res[c]["TOTAL"].mean(),
               "delta": d.mean(), "delta_sem": d.sem(), "up": int((d > 0).sum()),
               "n": len(d), "beats": bool(d.mean() > 2 * d.sem()) if d.sem() > 0 else False}
        for m in ("M1", "M2", "M3", "M4", "M5", "M6"):
            row[f"d{m}"] = (res[c][m].reindex(r.index) - res[tuple(ref)][m]).mean()
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["config", "delta"])
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cheap")
    ap.add_argument("--base", default="A,B,C,D,E")
    ap.add_argument("--ref", default="A")
    ap.add_argument("--members", default=None, help="restrict to these (comma list)")
    ap.add_argument("--greedy", type=int, default=0)
    ap.add_argument("--combos", default="", help="semicolon-separated comma lists")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("VCELL_WORKERS", 12)))
    ap.add_argument("--folds", default=None, help="restrict folds: '1:CGD,3:BAH'")
    args = ap.parse_args()

    pool_dir = os.path.join(OUT, f"pool_{args.tag}")
    folds = FOLDS if args.folds is None else [
        (int(f.split(":")[0]), f.split(":")[1]) for f in args.folds.split(",")]
    have = {}
    for f in os.listdir(pool_dir):
        if f.endswith(".npy") and not f.endswith(".tmp.npy"):
            fold, member = f[:-4].split("__")
            have.setdefault(member, set()).add(fold)
    need = {f"{s}_{st}" for s, st in folds}
    members = sorted(m for m, fs in have.items() if need <= fs)
    if args.members:
        members = [m for m in args.members.split(",") if m in members]
    incomplete = sorted(m for m, fs in have.items() if not need <= fs)
    print(f"pool={pool_dir}  folds={len(folds)}  complete members ({len(members)}): {members}")
    if incomplete:
        print(f"  incomplete (skipped): {incomplete}")
    base = [m for m in args.base.split(",") if m]
    ref = (args.ref,)
    assert all(m in members for m in base + list(ref)), "base/ref members missing from pool"

    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    t0 = time.time()
    ev = Evaluator(pool_dir, folds, args.workers)

    print(f"\n=== 1. singles vs '{args.ref}' (paired, {len(folds)} orphan-free folds) ===")
    print(paired_table(ev, [(m,) for m in members], ref).round(5).to_string(index=False))

    print(f"\n=== 2. base ensemble {'+'.join(base)} vs '{args.ref}', and base minus each member ===")
    combos = [tuple(base)] + [tuple(m for m in base if m != x) for x in base]
    labels = {tuple(base): "BASE " + "+".join(base)}
    labels.update({tuple(m for m in base if m != x): f"BASE - {x}" for x in base})
    print(paired_table(ev, combos, ref, lambda c: labels.get(c, "+".join(c)))
          .round(5).to_string(index=False))

    print(f"\n=== 3. base + each candidate, paired against BASE ===")
    cands = [m for m in members if m not in base]
    combos = [tuple(base) + (m,) for m in cands]
    print(paired_table(ev, combos, tuple(base), lambda c: f"BASE + {c[-1]}")
          .round(5).to_string(index=False))

    if args.greedy:
        print(f"\n=== 4. greedy forward selection from BASE ({args.greedy} steps) ===")
        cur = tuple(base)
        for step in range(args.greedy):
            cands = [m for m in members if m not in cur]
            if not cands:
                break
            tab = paired_table(ev, [cur + (m,) for m in cands], cur, lambda c: c[-1])
            if tab.empty:
                break
            best = tab.iloc[0]
            ru = (f"(runner-up {tab.iloc[1].config} {tab.iloc[1].delta:+.5f})"
                  if len(tab) > 1 else "")
            print(f"  step {step+1}: best add = {best.config}  delta={best.delta:+.5f} "
                  f"sem={best.delta_sem:.5f} up={best.up}/{best.n}   {ru}")
            if best.delta <= 0:
                print("  no positive addition left; stop")
                break
            cur = cur + (best.config,)
        tab = paired_table(ev, [cur], ref, lambda c: "GREEDY " + "+".join(c))
        print(tab.round(5).to_string(index=False))
        tab = paired_table(ev, [cur], tuple(base), lambda c: "GREEDY vs BASE")
        print(tab.round(5).to_string(index=False))

    if args.combos:
        print(f"\n=== 5. explicit combos vs '{args.ref}' and vs BASE ===")
        combos = [tuple(x.split(",")) for x in args.combos.split(";") if x]
        print(paired_table(ev, combos, ref).round(5).to_string(index=False))
        print(paired_table(ev, combos, tuple(base), lambda c: "+".join(c) + "  vs BASE")
              .round(5).to_string(index=False))

    rows = []
    for c, df in ev.cache.items():
        for (s, st), r in df.iterrows():
            rows.append({"combo": "+".join(c), "seed": s, "strain": st, **r.to_dict()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"pool_{args.tag}_eval.csv"), index=False)
    ev.close()
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

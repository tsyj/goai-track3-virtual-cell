# -*- coding: utf-8 -*-
"""官方 val 镜像上按成员保存 (加性, booster) 两部分，供未见实体重定标的最终验证。

    VCELL_MEMBER=<名> python scripts/71_val_parts.py   # 单成员（便于并行）
"""
import os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import importlib.util
_s = importlib.util.spec_from_file_location("m55", os.path.join(ROOT, "scripts", "55_member_pool.py"))
m55 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m55)
from vcell.harness import build_fold
from vcell.models import PERT_FACTORS, ResidualBooster, UnifiedBackfit

OUT = os.path.join(ROOT, "results", "val_parts")
REAL = {"n_comp": 240, "n_estimators": 1600, "learning_rate": 0.015, "seeds": [0, 1, 2]}
name = os.environ["VCELL_MEMBER"]
cfg = m55.MEMBERS[name]
os.makedirs(OUT, exist_ok=True)
pa, pb = os.path.join(OUT, f"{name}_add.npy"), os.path.join(OUT, f"{name}_boost.npy")
if os.path.exists(pa) and os.path.exists(pb):
    print("exists", name); sys.exit(0)
t0 = time.time()
fo = build_fold()
lam = {**m55.BASE_LAM, **cfg.get("lam", {})}
batch = [m55.BY_NAME[a] for a in cfg.get("order", m55.CUR_ORDER)]
um = UnifiedBackfit(
    batch_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in batch],
    pert_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
    n_pass=cfg.get("n_pass", 6), fit_offset=cfg.get("fit_offset", True),
).fit(fo.meta, fo.Y_obs, fo.obs_mask)
P_add = um.predict()
boost = {**REAL, **cfg.get("booster_over", {})}
rb = ResidualBooster(n_jobs=int(os.environ.get("VCELL_LGB_THREADS", 16)), **boost)
rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
B = rb.predict().astype(np.float32)
np.save(pa + ".tmp.npy", P_add); os.replace(pa + ".tmp.npy", pa)
np.save(pb + ".tmp.npy", B); os.replace(pb + ".tmp.npy", pb)
print(f"{name} done {time.time()-t0:.0f}s")

"""复赛交付管线：把 configs/final.yaml 里冻结的配置变成可训练、可推理的两步。

设计要点（对应《虚拟细胞方向材料提交说明》四、五节）：

* **训练与推理分离**。``fit_member`` 只看 ``split_final == 'train'`` 的标签；它把拟合结果
  写成冻结产物（加性项表、booster 的成分基与测试行成分得分），``predict_member`` 只读这些
  产物和测试 metadata，不接触任何标签。

* **为什么产物里包含测试行的成分得分**。本模型是**转导式**的：官方把测试集的 metadata
  公开，模型在 train+test 的**行并集**上拟合设计矩阵，只有 train 的**标签**可见。因此
  设计矩阵（板号、菌株、化合物…的水平编码）在训练时就已完全确定，booster 在其上的输出
  是一个固定的 13,412×240 矩阵。冻结它是无损的，且比冻结 8,640 棵 LightGBM 森林
  （约 26 GB）现实得多。产生它的训练代码一并提交，可从头重建。

* **不含任何测试标签**。冻结产物里只有模型参数与设计矩阵编码；``data/quarantine`` 下的
  测试真值文件在 ``vcell/io.py`` 层被硬性拒绝读取。

Jiao Xinyuan 2026-09-02
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from .harness import protein_keep_mask
from .io import load_combined
from .models import BATCH_FACTORS, PERT_FACTORS, ResidualBooster, UnifiedBackfit

__all__ = ["load_config", "build_design", "fit_member", "save_member",
           "load_member", "predict_member", "compose", "sha256_of"]


# ---------------------------------------------------------------- 配置
def load_config(path):
    if path.endswith((".yml", ".yaml")):
        import yaml
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
    else:
        cfg = json.load(open(path, encoding="utf-8"))
    for m in cfg["ensemble"]["members"]:
        m.setdefault("order", None)
        m.setdefault("lam", {})
        m.setdefault("n_pass", cfg["model"]["n_pass"])
        m.setdefault("fit_offset", cfg["model"]["fit_offset"])
        m.setdefault("booster", {})
    return cfg


def _lam_table(cfg, member):
    """把 configs 里的 base_lambdas + 成员覆盖，展开成 {因子名: lambda}。"""
    lam = dict(cfg["model"]["base_lambdas"])
    mult = member.get("pert_mult")
    if mult:
        for a, _, l in PERT_FACTORS:
            lam[a] = l * float(mult)
    for k, v in (member.get("lam") or {}).items():
        lam[k] = float(v)
    return lam


def _factors(cfg, member):
    by_name = {a: (a, c, l) for a, c, l in BATCH_FACTORS}
    order = member.get("order") or [a for a, _, _ in BATCH_FACTORS]
    assert sorted(order) == sorted(by_name), f"{member['name']}: 因子顺序不是原集合的排列"
    lam = _lam_table(cfg, member)
    batch = [(a, c, lam.get(a, l)) for a, c, l in (by_name[x] for x in order)]
    pert = [(a, c, lam.get(a, l)) for a, c, l in PERT_FACTORS]
    return batch, pert


# ---------------------------------------------------------------- 设计矩阵
def build_design(cfg, train_meta=None, train_proteome=None, test_meta=None):
    """载入 train+test 行的并集与训练标签，并施加官方蛋白过滤。

    默认走 ``vcell.io.load_combined()``（读 data/input 下的官方文件）；传入显式路径时
    仅用于核对，实际读取仍由 io 层负责，以保证隔离规则生效。
    """
    P = load_combined()
    keep = protein_keep_mask(P.meta, P.X, cfg["data"]["protein_missing_max"])
    P.X = P.X[:, keep]
    P.proteins = P.proteins[keep]
    is_test = (P.meta["SET"] == "test").to_numpy()
    visible = (P.meta["split_final"] == cfg["data"]["train_split_value"]).to_numpy()
    assert visible.sum() == cfg["data"]["expect_train_rows"], \
        f"训练行数 {visible.sum()} != 配置声明的 {cfg['data']['expect_train_rows']}"
    assert keep.sum() == cfg["data"]["expect_n_proteins"], \
        f"蛋白数 {keep.sum()} != 配置声明的 {cfg['data']['expect_n_proteins']}"
    assert is_test.sum() == cfg["data"]["expect_test_rows"]
    return P, visible, is_test


# ---------------------------------------------------------------- 训练
def fit_member(cfg, member, P, visible, is_test, n_jobs=16, verbose=True):
    batch, pert = _factors(cfg, member)
    um = UnifiedBackfit(batch_factors=batch, pert_factors=pert,
                        n_pass=int(member["n_pass"]),
                        fit_offset=bool(member["fit_offset"])).fit(P.meta, P.X, visible)
    add = um.predict()
    boost = {**cfg["model"]["booster"], **(member.get("booster") or {})}
    rb = ResidualBooster(n_jobs=n_jobs, **boost).fit(P.meta, P.X, visible, add)
    Z = np.mean([np.column_stack([g.predict(rb._X) for g in ms]) for ms in rb.model_sets], 0)
    if verbose:
        print(f"    booster: {boost}  成分 {rb.V.shape[0]}  解释残差方差 {rb.explained:.3f}",
              flush=True)
    return {
        "name": member["name"],
        "terms": {k: v for k, v in um.terms.items()},
        "codes": {k: v.astype(np.int32) for k, v in um.codes.items()},
        "batch_names": [a for a, _, _ in batch],
        "pert_names": sorted(um.pert_names),
        "mu": um.mu, "mu_fallback": float(um.mu_fallback),
        "offset": um.offset,
        "V": rb.V.astype(np.float32),
        "Z_all": Z.astype(np.float32),          # 全部行的成分得分（后处理需要对照孔的预测）
        "booster_scale": float(rb.scale),
        "config": {"order": [a for a, _, _ in batch],
                   "lam": {a: float(l) for a, _, l in batch + pert},
                   "n_pass": int(member["n_pass"]),
                   "fit_offset": bool(member["fit_offset"]), "booster": boost},
    }


def save_member(art, run_dir):
    d = os.path.join(run_dir, "members", art["name"])
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "additive.npz"),
                        mu=art["mu"], offset=art["offset"],
                        **{f"T__{k}": v for k, v in art["terms"].items()},
                        **{f"C__{k}": v for k, v in art["codes"].items()})
    np.savez_compressed(os.path.join(d, "booster.npz"), V=art["V"], Z_all=art["Z_all"])
    json.dump({"name": art["name"], "mu_fallback": art["mu_fallback"],
               "batch_names": art["batch_names"], "pert_names": art["pert_names"],
               "booster_scale": art["booster_scale"], "config": art["config"]},
              open(os.path.join(d, "member.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return d


def load_member(run_dir, name):
    d = os.path.join(run_dir, "members", name)
    a = np.load(os.path.join(d, "additive.npz"))
    b = np.load(os.path.join(d, "booster.npz"))
    j = json.load(open(os.path.join(d, "member.json"), encoding="utf-8"))
    return {"name": name, "mu": a["mu"], "offset": a["offset"],
            "terms": {k[3:]: a[k] for k in a.files if k.startswith("T__")},
            "codes": {k[3:]: a[k] for k in a.files if k.startswith("C__")},
            "V": b["V"], "Z_all": b["Z_all"], **j}


def predict_member(art, is_test, boost_mult=None):
    """兼容入口：只算测试行。见 predict_rows。"""
    idx = np.where(is_test)[0]
    return predict_rows(art, idx, boost_mult=boost_mult)


def predict_rows(art, idx, boost_mult=None):
    """重建该成员在测试行上的预测：加性部分 + booster 成分重构。

    ``boost_mult``：可选的逐测试行 booster 乘子（形状 = 测试行数）。用于「未见实体行的
    booster 重定标」：树模型对训练中未出现过的类别水平输出系统性偏弱（样本落入分裂默认侧、
    预测塌向其余类别的均值），对零标签菌株的行按冻结在配置里的 k 放大补偿。
    k 在六个无孤儿内层折上标定、在官方 val 镜像上验证，见 scripts/experiments/69,70。"""
    idx = np.asarray(idx)
    mu = art["mu"]
    out = np.tile(mu, (len(idx), 1)).astype(np.float32)
    for name in art["batch_names"] + list(art["pert_names"]):
        out += art["terms"][name][art["codes"][name][idx]]
    out += art["offset"][idx][:, None]          # 留出行的 offset 恒为 0
    boost = art["booster_scale"] * (art["Z_all"][idx] @ art["V"])
    if boost_mult is not None:
        boost = boost * np.asarray(boost_mult, np.float32)[:, None]
    out += boost
    return np.where(np.isfinite(out), out, art["mu_fallback"]).astype(np.float32)


def compose(preds, weights=None):
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, float)
    acc = None
    for p, wi in zip(preds, w):
        acc = p.astype(np.float64) * wi if acc is None else acc + p.astype(np.float64) * wi
    return (acc / w.sum()).astype(np.float32)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def effect_expansion(P, meta, beta, tau, row_mask=None):
    """大效应非线性扩张（后处理，作用于集成均值）。

    收缩估计把大扰动效应压得偏小；对模型隐含的效应 D = P − C（C = 同上下文对照孔预测
    的均值）做 h(D) = D·(1 + β·min(|D|/τ, 1)²)：小效应不动，|D| ≥ τ 的效应放大 (1+β)。
    只改处理孔，不改对照孔；``row_mask`` 限定作用行（采纳：只作用于零标签菌株的行——
    可见菌株的行已校准到噪声地板，扩张只添噪；零标签菌株的行系统性欠离散）。
    标定：六无孤儿内层折 +0.0020±0.0002（6/6，9.8×sem），官方 val 镜像 0.5032→0.5050。
    见 scripts/experiments/77, 80, 83, 84, 85。"""
    P = np.asarray(P, np.float32)
    ctrl = meta["is_control"].to_numpy()
    ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        rows = df.index[df.ctx == c].to_numpy()
        C[rows] = P[g.i.to_numpy()].mean(0); has[rows] = True
    D = np.where(has[:, None], P - C, 0.0)
    gmul = 1.0 + float(beta) * np.minimum(np.abs(D) / float(tau), 1.0) ** 2
    apply = has & ~ctrl
    if row_mask is not None:
        apply = apply & np.asarray(row_mask, bool)
    out = np.where(apply[:, None], C + D * gmul, P).astype(np.float32)
    return out, int(has.sum()), int(apply.sum())

"""Build the demo page (demo/README.md + PNGs) from cached results -- no refitting.

Uses results/val_mirror_*.csv (official val mirror, held-out strain BAI),
results/pool_val/*.npy (member predictions on that mirror, from 59) and
results/pool_real_eval.csv (paired inner-fold deltas, from 56).

    python scripts/61_demo_report.py

Jiao Xinyuan 2026-08-16
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import build_fold  # noqa: E402

RES = os.path.join(ROOT, "results")
DEMO = os.path.join(ROOT, "demo")
os.makedirs(DEMO, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans CJK KR", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ens = pd.read_csv(os.path.join(RES, "val_mirror_ensembles.csv"), index_col=0)
mem = pd.read_csv(os.path.join(RES, "val_mirror_members.csv"), index_col=0)
single = ens.loc["A alone (v4 single, online until 20:16)"]
five = ens.loc["A-E 5-member (53/54 design)"]
ship = ens.loc["SHIPPED 12-member (2 orders x 6 lambda points)"]

# ---- fig 1: model ladder on the official val mirror
ladder = [("随机下限（打乱真值）", 0.197), ("蛋白均值基线", 0.240), ("对照孔 板×菌株 基线", 0.389),
          ("统一可加模型：只有批次项", 0.463), ("可加模型：批次 + 扰动", 0.478),
          ("+ 残差 GBDT（8-10）", 0.488), ("+ booster/层级/λ 重调（单配置）", round(single.TOTAL, 4)),
          ("12 成员集成（提交版）", round(ship.TOTAL, 4)), ("oracle：批次 + 真值 Δ（不可达）", 0.816)]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
names = [n for n, _ in ladder]
vals = [v for _, v in ladder]
colors = ["#bbbbbb"] * 6 + ["#4c72b0", "#dd8452", "#bbbbbb"]
ax.barh(range(len(vals)), vals, color=colors)
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 0.9)
ax.set_xlabel("官方 val 镜像六模块加权总分（留出菌株 BAI）")
ax.set_title("模型阶梯：分数主要来自批次结构，扰动建模与集成再各加一层")
fig.tight_layout()
fig.savefig(os.path.join(DEMO, "fig1_ladder.png"), dpi=150)
plt.close(fig)

# ---- fig 2: per-split FC and modules, single vs 12-member
splits = ["FC[chem_only]", "FC[strain_only]", "FC[both]", "FC[time]"]
mods = ["M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)", "M5_bt(10%)", "M6_DEP(5%)"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, cols, title in ((axes[0], splits, "各划分的 Δ 相关（PCC）"), (axes[1], mods, "六个评分模块")):
    x = np.arange(len(cols))
    w = 0.27
    ax.bar(x - w, single[cols].values, w, label="单配置 0.4992", color="#4c72b0")
    ax.bar(x, five[cols].values, w, label="5 成员 0.5001", color="#8da0cb")
    ax.bar(x + w, ship[cols].values, w, label="12 成员（提交）0.5032", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("FC[", "").replace("]", "").replace("(", "\n(") for c in cols], fontsize=8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("官方 val 镜像：集成的增益集中在未见菌株划分（strain_only）与 M2/M4")
fig.tight_layout()
fig.savefig(os.path.join(DEMO, "fig2_splits_modules.png"), dpi=150)
plt.close(fig)

# ---- fig 3: strain-early vs current order, member by member
pairs = [("A", "F_strain_early", "plate 2, pert×4"), ("B", "FB_early_plate1", "plate 1"),
         ("C", "FC_early_plate4", "plate 4"), ("D", "FD_early_pert2", "pert×2"),
         ("E", "FE_early_pert8", "pert×8"), ("E16_pert16", "FE16_early_pert16", "pert×16")]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(pairs))
w = 0.38
cur = [mem.loc[a, "TOTAL"] for a, _, _ in pairs]
early = [mem.loc[b, "TOTAL"] for _, b, _ in pairs]
ax.bar(x - w / 2, cur, w, label="当前顺序（plate 先于 strain）", color="#4c72b0")
ax.bar(x + w / 2, early, w, label="strain 先于 plate", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels([p for _, _, p in pairs])
ax.set_ylim(min(cur) - 0.004, max(early) + 0.003)
ax.set_ylabel("val 镜像总分")
ax.set_title("同一 λ 点上，strain 提前到 plate 之前拟合的成员全部更好（+0.002~+0.004）")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(DEMO, "fig3_order.png"), dpi=150)
plt.close(fig)

# ---- fig 4: delta_pred vs delta_true for held-out-strain samples (12-member average)
fo = build_fold()
members = ["A", "B", "C", "D", "E", "E16_pert16", "F_strain_early", "FB_early_plate1",
           "FC_early_plate4", "FD_early_pert2", "FE_early_pert8", "FE16_early_pert16"]
acc = None
for m in members:
    p = np.load(os.path.join(RES, "pool_val", f"{m}.npy")).astype(np.float64)
    acc = p if acc is None else acc + p
P = acc / len(members)
sc = fo.scorer
dP = P - sc._pred_control(P)
dT = sc.D_true
rows = np.where(fo.meta["split_final"].to_numpy() == "val_strain_only")[0]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
rng = np.random.default_rng(0)
for ax, i in zip(axes, rng.choice(rows, 3, replace=False)):
    ok = np.isfinite(dP[i]) & np.isfinite(dT[i])
    r = np.corrcoef(dP[i][ok], dT[i][ok])[0, 1]
    ax.scatter(dT[i][ok], dP[i][ok], s=3, alpha=0.35, color="#4c72b0")
    lim = np.nanpercentile(np.abs(dT[i][ok]), 99.5)
    ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    m = fo.meta.iloc[i]
    ax.set_title(f"{m['Strains']} · {m['compound']} · {m['pert_time']}min\nPCC(Δ_pred, Δ_true) = {r:.3f}", fontsize=9)
    ax.set_xlabel("Δ_true = y − y_control (log2)")
axes[0].set_ylabel("Δ_pred")
fig.suptitle("未见菌株（BAI）样本上的扰动效应预测：逐蛋白 Δ 散点（12 成员平均）", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(DEMO, "fig4_delta_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- inner-fold paired table from pool_real_eval.csv
ev = pd.read_csv(os.path.join(RES, "pool_real_eval.csv"))
piv = ev.pivot_table(index=["seed", "strain"], columns="combo", values="TOTAL")
AE = "A+B+C+D+E"
F5 = "F_strain_early+FB_early_plate1+FC_early_plate4+FD_early_pert2+FE_early_pert8"
c10 = AE + "+" + F5
c12 = "A+B+C+D+E+E16_pert16+" + F5 + "+FE16_early_pert16"
def paired(a, b):
    d = (piv[a] - piv[b]).dropna()
    return f"{d.mean():+.5f} ± {d.sem():.5f}（{int((d > 0).sum())}/{len(d)} 折）"
tab = [("5 成员 λ 邻域集成 vs 单配置 A", paired(AE, "A")),
       ("+ strain-early 五成员（10 成员）vs 单配置", paired(c10, "A")),
       ("12 成员（提交版）vs 单配置", paired(c12, "A")),
       ("12 成员 vs 5 成员", paired(c12, AE)),
       ("12 成员 vs 10 成员", paired(c12, c10))]

md = f"""# Demo · 从批次结构到扰动响应：分层收缩回填与残差提升的虚拟酵母模型

GOAI 2026 赛道三 · 方向一（虚拟酵母扰动蛋白质组预测）。本页由 `scripts/61_demo_report.py` 从缓存结果生成，
所有数字可用仓库代码复现（见 [`REPRODUCE.md`](../REPRODUCE.md)）。

## 1. 任务与方案一句话

预测 4,454 个测试样本 × 5,243 个蛋白的 log2 蛋白质组，覆盖未见化合物 / 未见菌株 / 双重未知 / 时间插值。
方案把预测拆成 **批次基线 + 扰动效应** 两部分，用**分层收缩回填**（`vcell/models.py::UnifiedBackfit`）
在 log2 空间统一拟合，再用 **残差 LightGBM**（`ResidualBooster`，残差 PCA 到 240 维后逐维一棵树）吃高阶交互；
最终提交是 **12 个近优配置的等权集成**（两种因子拟合顺序 × 六个 λ 邻域点）。
只用 `split_final=='train'` 的 5,920 行标签拟合；测试集真值文件被隔离，代码层禁读。

```python
um = UnifiedBackfit(batch_factors=BATCH_FACTORS, pert_factors=PERT_FACTORS, n_pass=6).fit(meta, Y_obs, use)
P  = um.predict()                                             # 批次结构 + 扰动效应（可加）
rb = ResidualBooster(n_comp=240, n_estimators=1600, learning_rate=0.015, seeds=[0, 1, 2]).fit(meta, Y_obs, use, P)
Y_hat = P + rb.predict()                                      # 一个集成成员
```

## 2. 分数从哪里来（官方 val 镜像，留出菌株 BAI，只评一次）

![ladder](fig1_ladder.png)

| | TOTAL | FC[chem_only] | FC[strain_only] | FC[both] | FC[time] |
|---|---:|---:|---:|---:|---:|
| 单配置 | {single.TOTAL:.4f} | {single['FC[chem_only]']:.4f} | {single['FC[strain_only]']:.4f} | {single['FC[both]']:.4f} | {single['FC[time]']:.4f} |
| 5 成员 λ 邻域集成 | {five.TOTAL:.4f} | {five['FC[chem_only]']:.4f} | {five['FC[strain_only]']:.4f} | {five['FC[both]']:.4f} | {five['FC[time]']:.4f} |
| **12 成员集成（提交版）** | **{ship.TOTAL:.4f}** | {ship['FC[chem_only]']:.4f} | **{ship['FC[strain_only]']:.4f}** | {ship['FC[both]']:.4f} | {ship['FC[time]']:.4f} |

![splits](fig2_splits_modules.png)

## 3. 为什么是这 12 个成员：一个被否决的"替代方案"是最好的集成成员

把 `strain` 因子提前到 `plate` 之前拟合，作为**替代**时只在 3/6 折上更好（未过线）；
但作为**集成成员**它 6/6 折过线——它的偏差方向与当前顺序不同，正是集成需要的多样性。
在官方 val 镜像上，同一 λ 点的 strain-early 成员**全部**优于当前顺序的成员：

![order](fig3_order.png)

内层六折（与官方结构一致的无孤儿 plate 折，逐折配对，真配置 booster）：

| 比较 | 配对 delta ± sem |
|---|---|
""" + "\n".join(f"| {a} | {b} |" for a, b in tab) + f"""

## 4. 未见菌株上的扰动效应长什么样

![scatter](fig4_delta_scatter.png)

## 5. 评测审计（方法的另一半）

`vcell/metrics.py` 忠实复现了六个评分模块；`scripts/09_metric_audit.py`、`27_official_baselines.py`、
`30_calibrate_official.py` 用组委会公布的基线数字校准了对齐 / 过滤 / 尺度 / 对照匹配
（蛋白均值基线的 Global R² 与官方最大偏差 0.3%）。审计发现：**同一份预测在不同合理评分读法下总分跨度 0.41–0.57**，
比一切建模改进都大——相关口径问题列在 [`docs/OPEN_QUESTIONS.md`](../docs/OPEN_QUESTIONS.md)。

## 6. 复现

```bash
pip install -r requirements.txt            # Python 3.9, numpy/pandas/scipy/lightgbm, CPU only
# 官方 3 个输入文件放到 data/input/ 后：
for m in A B C D E E16_pert16 F_strain_early FB_early_plate1 FC_early_plate4 FD_early_pert2 FE_early_pert8 FE16_early_pert16; do
  python scripts/57_predict_member.py $m; done
python scripts/58_compose_submission.py --members A,B,C,D,E,E16_pert16,F_strain_early,FB_early_plate1,FC_early_plate4,FD_early_pert2,FE_early_pert8,FE16_early_pert16 --out submission/_candidates/ens12/prediction.csv
python scripts/60_full_feature_contract.py --src submission/_candidates/ens12/prediction.csv --out submission/prediction.csv
```
"""
open(os.path.join(DEMO, "README.md"), "w").write(md)
print("wrote demo/README.md and 4 figures")

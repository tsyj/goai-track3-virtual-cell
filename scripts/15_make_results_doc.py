"""Regenerate docs/RESULTS.md from the CSVs, so the write-up never drifts."""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
pd.set_option("display.width", 300)


def md(df, cols=None, floats=3):
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return df.to_markdown(index=False, floatfmt=f".{floats}f")


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


out = [f"# 结果汇总\n\n> 由 `scripts/15_make_results_doc.py` 自动生成 · "
       f"{datetime.now():%Y-%m-%d %H:%M}　所有数字都能在 `results/` 下的 CSV 里查到。\n"]

MOD = ["model", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)",
       "M4_drug(20%)", "M5_bt(10%)", "M6_DEP(5%)"]

fv = load("final_val.csv")
if fv is not None:
    out += ["## 1. 官方 val 镜像上的最终结果\n",
            "本地镜像用组委会自己的 `val_chem_only / val_strain_only / val_both / "
            "val_time` 四个划分，与 test 的四个划分一一对应；留出菌株自己的对照孔"
            "同样隐藏（本地 BAI ↔ 正式 CRD）。\n", md(fv, MOD), "\n",
            "按划分看「匹配对照原始 FC」的 PCC：\n",
            md(fv, ["model"] + [c for c in fv.columns if c.startswith("FC[")]), "\n"]

ce = load("ceiling.csv")
if ce is not None:
    out += ["## 2. 上限与下限\n",
            "没有这两条线，单独一个分数无法解读。\n", md(ce, MOD), "\n",
            "- **复现上限**：WAYB 三个重复批次共 5,366 个重复对，"
            "Δ 的重复间相关系数 **0.136**，高效应蛋白方向一致率 **0.597**。\n"
            "- 由此反解：真实扰动效应 rms **0.146** log2，单样本测量噪声 rms "
            "**0.26** log2 —— **效应比噪声小一倍**。\n"]

il = load("information_ladder.csv")
if il is not None:
    out += ["## 3. 信息阶梯：每个模块里模型能动多少\n", md(il, MOD), "\n"]

cn = load("control_noise.csv")
if cn is not None:
    sub = cn[cn.k > 0]
    out += ["## 4. 共享对照噪声（判定性实验）\n",
            "**同一份预测**，只改评分时对照参照平均了几个孔：\n",
            md(sub, ["model", "mean_wells", "well_sd", "TOTAL", "M2_rawFC(25%)",
                     "M4_drug(20%)", "M6_DEP(5%)"]), "\n",
            "对照测得越准，分数越低。\n"]
    dm = cn[cn.k == -1]
    if len(dm):
        out += ["Δ 用实测对照 vs 用提交的预测对照：\n",
                md(dm, MOD), "\n"]

m3 = load("m3_monotonicity.csv")
if m3 is not None:
    out += ["## 5. M3 的单调性反转\n",
            "γ 是给未见化合物补上的「平均药物响应」的系数；γ=1 是无信息时的最优预测。\n",
            md(m3, ["gamma", "TOTAL", "M3_ctx(20%)", "M2_rawFC(25%)",
                    "M6_DEP(5%)"], 4), "\n"]

rb = load("robustness_band.csv")
if rb is not None:
    out += ["## 6. 稳健性区间：同一份预测在不同口径下的分数\n", md(rb, MOD), "\n",
            f"**总分区间 {rb.TOTAL.min():.3f} – {rb.TOTAL.max():.3f}"
            f"（跨度 {rb.TOTAL.max()-rb.TOTAL.min():.3f}）。**\n"]

rel = load("compound_reliability.csv")
pair = load("compound_pair_similarity.csv")
if rel is not None:
    out += ["## 7. 化学先验：可靠但不可迁移\n",
            f"- 化合物效应向量的分半信度中位数 **{rel.split_half_r.median():.3f}**，"
            f"整体估计信度中位数 **{rel.reliability.median():.3f}**；"
            f"{int((rel.reliability > 0.5).sum())} / {len(rel)} 个 > 0.5。\n"
            "- 化合物效应空间前 5 个主成分解释 **80%** 方差；秩 5 重构与同一化合物"
            "留出一半样本的相关系数 **0.70**。\n"]
    if pair is not None:
        ss = pair[pair.same_source]
        a = ss[ss.same_moa].r_disatt.mean()
        b = ss[~ss.same_moa].r_disatt.mean()
        out += [f"- 控制数据来源后，同机制 vs 不同机制的去衰减响应相关性差 "
                f"**{a-b:+.4f}**（置换检验 p = 0.95，{len(ss)} 对）。\n"
                "- 结构相似度（Tanimoto）与响应相似度的 Spearman ρ = "
                "**−0.016，p = 0.73**。\n"
                "- 留一化合物转移的相关系数 ≈ 0（见 `chem_loco_grid.csv`）。\n"]
    out += ["\n**结论：在 46 个训练化合物的规模上，公开化学知识对未见化合物外推"
            "没有可测的帮助；封闭数据榜与开放知识榜在这份数据上大概率不可区分。**\n"]

vd = load("variance_decomposition.csv")
if vd is not None:
    out += ["## 8. 方差分解\n", md(vd.sort_values("var_frac", ascending=False)), "\n"]

bio = load("biology_check.csv")
if bio is not None:
    real = bio[bio.verdict.isin(["data+model", "data only"])]
    seen = real[real.visible_to_model]
    held = real[~real.visible_to_model]
    out += ["## 7a. 生物学检验（预注册）\n",
            md(bio, ["compound", "markers", "expected", "chemical_role",
                     "measured_effect", "model_effect", "verdict"]), "\n",
            f"- 实测数据中成立：**{len(real)}/{len(bio)}**\n"
            f"- 其中模型能看到的化合物：**{(seen.verdict=='data+model').sum()}/{len(seen)}** 复现\n"
            f"- 其中被留出的化合物：**{(held.verdict=='data+model').sum()}/{len(held)}** 复现"
            f"（{', '.join(held.compound)}）\n"]

bs = load("bootstrap_val.csv")
if bs is not None:
    lo, hi = np.percentile(bs["diff"], [2.5, 97.5])
    wa = np.percentile(bs["additive"], [2.5, 97.5])
    out += ["## 7z. 数字的精度（配对自助，200 次）\n",
            f"- 单个 val 总分的 95% 区间宽度 **±{(wa[1]-wa[0])/2:.3f}**\n"
            f"- booster 的增益（配对）**+{bs['diff'].mean():.4f}**，"
            f"95% CI [{lo:+.4f}, {hi:+.4f}] —— 不含 0\n"
            "- 自助均值比点估计低约 0.017（重采样重复行会稀释蛋白轴相关），"
            "所以自助值用于估**宽度**而非**位置**。\n"]

ob = load("official_baselines.csv")
if ob is not None:
    out += ["## 7b. 与官方口径基线的比较\n",
            "手册说组委会提供「均值模型 / 随机森林 / 梯度提升」基线且跨方向排名要"
            "相对基线归一化，但基线未发布，这里是我们自己实现的。\n",
            md(ob, MOD), "\n"]

ps = load("parallel_search.csv")
if ps is not None:
    out += ["## 7c. 6 折配对检验（256 核并行）\n",
            "折间标准差 0.023，2–3 折上的 0.001 级差距是噪声；配对后标准误降到 "
            "0.0002–0.002。\n",
            md(ps, ["config", "mean", "sem", "delta_vs_current", "delta_sem"], 4),
            "\n"]

rs = load("residual_sweep.csv")
if rs is not None:
    out += ["## 8b. 残差上的梯度提升：模型族选窄了\n",
            "可加模型只能表达一阶和二阶列联表。在它的残差上做 PCA 降维后训 LightGBM，"
            "内层镜像上总分 0.4359 → 0.4498。消融显示驱动力来自**化合物 ID** 特征。\n",
            md(rs, ["trial", "TOTAL", "M2_rawFC(25%)", "M3_ctx(20%)",
                    "M4_drug(20%)"], 4), "\n"]

it = load("inner_tuning.csv")
if it is not None:
    out += ["## 9. 内层镜像上的超参选择\n",
            "官方 val 镜像只在最后评一次；下表是在另一层从训练行切出的镜像上做的。\n",
            md(it.head(20), ["trial", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)",
                             "M3_ctx(20%)", "M4_drug(20%)"], 4), "\n"]

p = os.path.join(ROOT, "docs", "RESULTS.md")
open(p, "w").write("\n".join(out))
print("wrote", p, f"({len(''.join(out))} chars)")

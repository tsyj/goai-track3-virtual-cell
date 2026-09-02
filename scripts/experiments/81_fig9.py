# -*- coding: utf-8 -*-
import os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "demo")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC","Noto Sans CJK JP","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GREY = "#4c72b0", "#dd8452", "#999999"
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
# (1) 变换形状
ax = axes[0]; x = np.linspace(-2.5, 2.5, 400)
for beta, tau, c in [(0.3, 1.25, ORANGE)]:
    ax.plot(x, x * (1 + beta * np.minimum(np.abs(x) / tau, 1) ** 2), color=c, lw=2, label=f"h(D)，β={beta} τ={tau}")
ax.plot(x, x, color=GREY, ls="--", label="不变换"); ax.axvline(1.25, color=GREY, lw=.6); ax.axvline(-1.25, color=GREY, lw=.6)
ax.set_xlabel("模型隐含效应 D = P − C（log2）"); ax.set_ylabel("扩张后 h(D)"); ax.set_title("① 只放大大效应：|D|≥τ 时 ×(1+β)", fontsize=10.5); ax.legend(fontsize=8.5); ax.grid(alpha=.3)
# (2) 内层折 vs val
ax = axes[1]
d = pd.read_csv(os.path.join(RES, "expand_inner_fine.csv")); d = d[d.tau == 1.0]
piv = d.pivot_table(index=["seed","strain"], columns="beta", values="TOTAL"); bs = sorted(piv.columns)
ax.errorbar(bs, [(piv[b]-piv[0.0]).mean() for b in bs], yerr=[2*(piv[b]-piv[0.0]).sem() for b in bs], marker="o", color=BLUE, capsize=3, label="内层六折（12 成员真配置，2×sem）")
v = pd.read_csv(os.path.join(RES, "expand_val.csv")); v1 = v[v.tau == 1.0].sort_values("beta"); b0 = float(v.loc[v.beta==0, "TOTAL"].iloc[0])
ax.plot(v1.beta, v1.TOTAL - b0, marker="s", color=ORANGE, label="官方 val 镜像（留出 BAI）")
ax.axhline(0, color="k", lw=.8); ax.set_xlabel("β（τ=1.0）"); ax.set_ylabel("总分变化"); ax.set_title("② 两个口径一致：+0.0005~0.0007，6/6", fontsize=10.5); ax.legend(fontsize=8.5); ax.grid(alpha=.3)
# (3) 模块分解（val）
ax = axes[2]
mods = [("M2_rawFC(25%)","M2"),("M4_drug(20%)","M4"),("M6_DEP(5%)","M6"),("M1_abs(20%)","M1")]
row = v[(v.beta==0.3)&(v.tau==1.25)].iloc[0]; base = v[v.beta==0].iloc[0]
vals = [float(row[m]-base[m]) for m,_ in mods]
ax.bar([n for _,n in mods], vals, color=[ORANGE if x>0 else GREY for x in vals])
for i, x in enumerate(vals): ax.text(i, x + (0.0002 if x>=0 else -0.0006), f"{x:+.4f}", ha="center", fontsize=9)
ax.axhline(0, color="k", lw=.8); ax.set_title("③ 代价结构（val，β=0.3 τ=1.25）：Δ 类模块升，M1 微付", fontsize=10.5); ax.set_ylabel("模块分变化")
fig.suptitle("复赛新增（采纳）：大效应非线性扩张——把收缩估计压掉的尾部还回来", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_expansion.png"), dpi=150, bbox_inches="tight"); print("fig9_expansion.png")

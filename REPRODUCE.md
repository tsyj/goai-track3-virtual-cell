# 复现说明（初赛源代码提交 · 2026-08-16）

本包是 GOAI 2026 赛道三 · 方向一（虚拟酵母扰动蛋白质组预测）的**训练与推理源代码**。
按 2026 年 8 月修订版参赛手册要求：随机种子与关键参数在代码中体现，外部资源注明来源与版本，
不含运行环境打包。目标是**代码完整、流程可复现**。

---

## 1. 环境

| 项 | 值 |
|---|---|
| Python | 3.9.23（conda） |
| numpy | 2.0.2 |
| pandas | 2.3.1 |
| scipy | 1.13.1 |
| lightgbm | 4.6.0 |
| pyarrow | 21.0.0（缓存用） |
| rdkit | 2025.09.2（**仅** `vcell/chem.py` / 化学相似度实验用，最终模型不需要） |
| 硬件 | 纯 CPU；最终提交的 12 个集成成员各约 30 min（16 线程），可并行 |

```bash
pip install -r requirements.txt
```

## 2. 数据放置

把组委会发放的官方文件放到 `data/input/`（文件名保持原样）：

```
data/input/WAYB_WAYC_metadata_train_val(1).csv
data/input/WAYB_WAYC_metadata_test(1).csv
data/input/WAYB_WAYC_proteome_raw_train_val.csv
```

**`WAYB_WAYC_proteome_raw_test.csv`（测试集真值）不需要、也不能被代码读到。**
我们把它移到 `data/quarantine/`；`vcell/io.py::load_proteome` 在被请求 `test` 时直接抛异常，
训练、模型选择与推理代码在物理上无法读到它。首次运行会自动在 `data/cache/` 建 log2 缓存。

## 3. 一键复现最终提交（`submission/prediction.csv`）

最终提交 = **12 个近优配置的等权平均**，每个成员 = 结构化可加模型 `UnifiedBackfit`
＋残差 LightGBM `ResidualBooster`（240 成分 / 1600 树 / lr 0.015 / 3 个种子），只用
`split_final == 'train'` 的 5,920 行标签拟合。成员定义在 `scripts/55_member_pool.py::MEMBERS`：

| 成员 | 因子拟合顺序 | λ 邻域点 |
|---|---|---|
| A / B / C / D / E / E16 | 当前顺序（source→instrument→plate→plate×strain→strain→strain×…） | plate 2 pert×4（采纳点）/ plate 1 / plate 4 / pert×2 / pert×8 / pert×16 |
| F / FB / FC / FD / FE / FE16 | strain 提前到 plate 之前 | 同上六点 |

```bash
PY=python
# (1) 12 个成员的测试集预测（各约 30 min，互相独立，可并行；输出 submission/members/<m>.npy + .json）
for m in A B C D E E16_pert16 F_strain_early FB_early_plate1 FC_early_plate4 FD_early_pert2 FE_early_pert8 FE16_early_pert16; do
  VCELL_LGB_THREADS=16 $PY scripts/57_predict_member.py $m
done
# (2) 等权平均 → 4,422 列（train 行缺失率 <0.80 的蛋白）
$PY scripts/58_compose_submission.py \
  --members A,B,C,D,E,E16_pert16,F_strain_early,FB_early_plate1,FC_early_plate4,FD_early_pert2,FE_early_pert8,FE16_early_pert16 \
  --out submission/_candidates/ens12/prediction.csv
# (3) 补齐到官方蛋白质组文件的全部 5,243 列（原始列顺序）；4,422 列逐位不变，
#     其余 821 个低覆盖蛋白由配置 A 的可加模型（全蛋白拟合）给出
$PY scripts/60_full_feature_contract.py \
  --src submission/_candidates/ens12/prediction.csv --out submission/prediction.csv
```

**随机性与确定性**：可加模型完全确定；LightGBM 种子为 `seeds=[0,1,2]`，每个成分 j 的
`random_state = seed + j`（`vcell/models.py::ResidualBooster`）。LightGBM 在不同线程数下有
约 1e-3 log2 量级的数值差异（我们实测同一配置 32 线程 vs 16 线程：逐元素 rms 差 0.003，相关 0.9999995），
不影响任何结论。

提交文件：4,454 行（`sample_ID`，与测试元数据顺序一致）× 5,243 列（与官方蛋白质组文件表头逐字节一致，
含带引号的 `ARG5,6` / `DUR1,2`），log2 尺度，无 NA/inf。`submission/manifest.json` 记录了每个成员的
λ / 顺序 / sha 及选择依据。

## 4. 模型选择是怎么做的（可检查性）

- 评分模块的忠实复现：`vcell/metrics.py`；官方 val 镜像评估：`scripts/14_final_eval.py`、`scripts/59_val_mirror_ensemble.py`。
- 超参与结构在 **train 行内部再切的六个内层镜像**上用逐折配对检验选择（折间标准差 0.023，是要找的效应的十倍，
  非配对比较一律无效）；官方 `val_*` 划分只用于最终报告，从不参与选择。
  相关脚本：`scripts/26b_focused_search.py`、`33`–`53`、`analyze_paired.py`、`55_member_pool.py`、`56_pool_eval.py`。
- 三个与官方结构一致的判定要点已写进 `docs/METHOD.md`（§8）与 `docs/OPEN_QUESTIONS.md`：
  只用 train 标签、蛋白过滤只用 train 行、提交 log2。
- 采纳 / 否决台账：`results/best_config.json`；逐折原始结果：`results/*_raw.csv`、`results/pool_*_eval.csv`。
- 单元测试：`python -m pytest tests/ -q`。

## 5. 外部资源声明

**最终提交的模型不使用任何外部数据**——输入只有官方元数据字段（菌株、化合物、培养基、温度、
时间、数据来源、仪器、板号、孔位）与官方蛋白质组标签。

以下外部资源只在**已否决**的探索性实验中用过，不影响提交结果，且**不随本包分发**：

| 资源 | 用途 | 来源 / 版本 | 使用脚本 |
|---|---|---|---|
| 1,011 株酵母基因组项目：SNP/ORF 距离矩阵、基因存在/缺失、拷贝数、35 条件表型矩阵 | 未见菌株的"基因组相似度供体迁移"（结论：α=0 最优，否决） | Peter et al., *Nature* 2018；http://1002genomes.u-strasbg.fr/files/（`1011DistanceMatrixBasedOnSNPs.tab.gz` 等，2018 发布版） | `scripts/32_strain_prior_endtoend.py` |
| PubChem 化合物 SMILES / 分子式（PUG REST） | 未见化合物的结构相似度迁移（结论：负，否决） | https://pubchem.ncbi.nlm.nih.gov/rest/pug/…（2026-08-09 抓取，`data/chem/pubchem.json` 逐条记录了查询 URL） | `vcell/chem.py`, `scripts/05_chem_transfer.py` |
| 化合物作用机制类别（人工整理） | 同上 | 教科书药理学 / 酵母文献常识，见 `vcell/chem.py` 注释 | 同上 |

## 6. 目录

```
vcell/       io.py（加载 + 隔离）models.py（UnifiedBackfit / ResidualBooster）harness.py（折构造）
             metrics.py（六模块评分）design.py（对照匹配）chem.py nn.py viz.py（探索用）
scripts/     00–32 探索 / 上限 / 基线 / 审计；33–53 调参与结构实验；54–60 集成与提交生成
tests/       评测原语与管线的单元测试
docs/        METHOD.md RESULTS.md OPEN_QUESTIONS.md（方法、结果、向组委会提出的口径问题）
results/     best_config.json（配置与采纳/否决台账）、逐折原始结果 CSV、官方 val 镜像结果
submission_manifest.json   最终提交文件的成员清单与 sha
```

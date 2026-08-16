# 从批次结构到扰动响应：分层收缩回填与残差提升的虚拟酵母模型

GOAI 2026 · 赛道三 算法赛 · 方向一：虚拟酵母扰动蛋白质组预测 ｜ **Demo：[`demo/README.md`](demo/README.md)** ｜ **复现：[`REPRODUCE.md`](REPRODUCE.md)**

预测 4,454 个留出样本的完整 log2 蛋白质组向量（5,243 个蛋白），
覆盖三类泛化场景：未见化合物（S1）、未见菌株（S2）、双重未知（S3），外加时间插值（test_time）。

> **初赛提交（2026-08-16）**：复现最终 `submission/prediction.csv` 的完整步骤见 **`REPRODUCE.md`**。
> 最终模型 = 12 个近优配置（两种因子拟合顺序 × 六个 λ 邻域点）的等权集成，
> 每个成员 = 结构化可加模型 + 残差 LightGBM；官方 val 镜像（留出菌株 BAI）总分
> 单配置 0.4992 → 集成 **0.5032**（`results/val_mirror_ensembles.csv`）。
> 下面"快速开始"里的脚本与数字是 8-10 那一批探索时的记录（当时 0.481），保留作为过程记录。

---

## 快速开始

```bash
PY=/home/xinyuan/anaconda3/envs/numpy1/bin/python

$PY scripts/00_eda.py            # 数据结构、缺失、方差分解、菌株特异丢失
$PY scripts/02_ceiling.py        # 复现上限（WAYB 三次重复）与空模型下限
$PY scripts/08_tune_inner.py     # 内层镜像上选超参（不碰官方 val 划分）
$PY scripts/09_metric_audit.py   # 六个评分模块的审计
$PY scripts/13_control_noise.py  # 共享对照噪声的判定性实验
$PY scripts/20_residual_ml.py    # 残差上训 GBDT：判定"模型族是不是选窄了"
$PY scripts/21_residual_sweep.py # 成分数扫描 + 特征消融
$PY scripts/22_nn_and_ensemble.py # 第三个模型族：embedding MLP 与集成（均被否决）
$PY scripts/25_split_blend.py    # 未见菌株划分上 M2 与 M4 的取舍曲线
$PY scripts/26b_focused_search.py # 256 核并行：6 折配对检验所有取舍
$PY scripts/27_official_baselines.py # 自己实现官方口径的三个基线
$PY scripts/14_final_eval.py     # 在官方 val 镜像上跑一次最终评估
$PY scripts/10_predict_test.py   # 全量重训 → submission/prediction_*.csv
$PY scripts/28_selfaudit_stats.py # 评测原语单元测试 + 配对自助置信区间
$PY scripts/29_biology_check.py  # 10 条预注册的酵母生物学检验
$PY scripts/12_figures.py        # 出图

/home/xinyuan/anaconda3/envs/numpy1/bin/python -m pytest tests/ -q   # 12 个测试
```

依赖：numpy、pandas、scipy、matplotlib、lightgbm、rdkit（仅化学分析用）。
**无需 GPU**；全流程在 CPU 上约 40 分钟。

模型是两层：`UnifiedBackfit`（结构化可加，吃批次结构）+ `ResidualBooster`
（残差 PCA 到 96 维后每维一棵 LightGBM，吃高阶交互）。
官方 val 镜像总分 **0.481**（随机下限 0.197，oracle 上限 0.818；
自助重采样宽度 ±0.020，所以只报三位小数）。
相对最强的官方口径基线（梯度提升 0.422）**+14.1%**。

---

## 数据完整性：测试集真值已被隔离

官方数据包里的 `WAYB_WAYC_proteome_raw_test.csv` **含有全部测试样本的真值标签**，
与手册"真值标签保留、由组委会离线评分"的说明不符。

处理方式：

- 该文件已移入 `data/quarantine/`，权限 `400`；
- `vcell/io.py::load_proteome` 在被请求 `test` 时**直接抛异常**，训练与
  模型选择代码物理上无法读到它；
- 所有评估都在**组委会自己提供的 `val_*` 划分**上做——这四个划分与四个
  `test_*` 划分一一对应，且同样隐藏了留出菌株自己的对照孔
  （本地 BAI ↔ 正式 CRD）；
- 超参在**另一层内层镜像**（从 `train` 行里再切一次）上选，官方 val 镜像只评一次。

详见 `docs/OPEN_QUESTIONS.md`。

---

## 目录

```
vcell/            io.py 载入与缓存 · design.py 对照匹配 · metrics.py 六模块评分
                  models.py 统一可加模型 · chem.py 化学相似度 · harness.py 评测镜像
scripts/          00–14，编号即执行顺序
results/          所有实验的 CSV 与日志；figs/ 图
submission/       prediction_log2.csv / prediction_raw.csv / manifest.json
docs/             OPEN_QUESTIONS.md 待组委会确认 · METHOD.md 方法 · RESULTS.md 结果
data/input/       原始 CSV（已去掉泄漏文件）
data/quarantine/  泄漏的测试真值，只读、不读取
data/chem/        PubChem 解析结果（57 个化合物）
```

## 许可与外部资源

代码 MIT。外部资源仅两项，均为公开数据：
PubChem PUG REST（化合物 SMILES，`scripts/04_fetch_chem.py` 记录了每条的请求 URL）、
RDKit 2025.09.2（Morgan 指纹）。作用机制分类表是人工整理的，逐条列在
`vcell/chem.py::MOA` 中以便审查。官方数据集不再分发。

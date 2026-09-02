# 虚拟细胞方向 · 复赛代码与复现材料

| | |
|---|---|
| **作品编号** | 以官网提交记录为准（每赛段最多 3 次，最后一次为评审版本） |
| **赛道 / 方向** | 赛道三 前沿探索 AI for Research · 算法赛题 · 方向一 虚拟细胞（AIVC） |
| **队伍** | 有枣儿 |
| **作品** | 从批次结构到扰动响应：分层收缩回填与残差提升的虚拟酵母模型 |
| **最终模型** | 结构化 12 成员集成：分层收缩回填（`UnifiedBackfit`）＋ 残差梯度提升（`ResidualBooster`）；自评镜像总分 0.5032（官方 val 划分口径，最终评审以组委会内部集为准） |
| **prediction.csv SHA256** | `c6750c9796d9faf4c898cf2465ed28a3ed7b0da88daacb7d21155b37c413c6c7` |
| **代码版本** | git tag `v2.0-semifinal` ／ commit `67450574b0fe` ／ 仓库 https://github.com/tsyj/goai-track3-virtual-cell |
| **配置 hash** | `configs/final.yaml` SHA256 `b71d5dfea3a03c676b381d0b0e8de202…` |
| **负责人** | 孙丽敏 · jxy23@mails.tsinghua.edu.cn（与官网报名账号一致） |
| **已知限制** | 见第 6 节 |

---

## 1. 三条主命令

三条命令可直接复制执行。默认从 `data/input/` 读取官方文件；数据放在别处时设
`VCELL_DATA_ROOT=<含 data/input 的目录>`。

```bash
pip install -r requirements.txt

# 1) 构建外部特征 / embedding —— 本作品【无需执行】
#    最终模型不使用任何外部数据。该命令会做静态扫描 + 正向验证，自证这一点。
python scripts/build_embeddings.py --check

# 2) 从头训练最终模型（12 个集成成员）
python scripts/train.py \
    --metadata "data/input/WAYB_WAYC_metadata_train_val(1).csv" \
    --proteome "data/input/WAYB_WAYC_proteome_raw_train_val.csv" \
    --config   configs/final.yaml \
    --output-dir runs/final

# 3) 冻结模型推理，生成 prediction.csv（写盘后自动执行第 4 步格式自检）
python scripts/predict.py \
    --metadata "data/input/WAYB_WAYC_metadata_test(1).csv" \
    --run-dir  runs/final \
    --output   prediction.csv

# 4) 提交前格式自检（行列、尺度、有限性、列序）
python scripts/validate_submission.py --prediction prediction.csv
```

**冒烟测试（约 3 分钟，用于先确认环境可用）**

```bash
bash scripts/smoke_test.sh
```

它用一个成员、缩减的 booster 跑通「训练 → 推理 → 校验」全链路，不产生正式结果。

### 资源与耗时（实测，amax：Intel 256 核 / 503 GB / 无 GPU）

| 步骤 | 耗时 | 峰值内存 | 说明 |
|---|---|---|---|
| 命令 1 | < 5 s | < 1 GB | 只做核验 |
| 命令 2（12 成员，串行） | **约 6 小时** | 约 40 GB | 每成员约 30–40 min；成员互相独立，可并行：`--members <名>` 每个成员一个进程，**全部结束后执行 `python scripts/train.py --finalize --output-dir runs/final` 汇总 run.json**（子集训练不会写 run.json，防止并行互相覆盖）。11 路并行实测约 40 min |
| 命令 3 | 约 1 min | 约 12 GB | 只做加性项重建 + 成分重构 |
| 首次运行额外开销 | 约 2 min | — | 把 8,958×5,243 的原始 CSV 转成 log2 缓存（`data/cache/`） |

无需 GPU。`runs/final` 约 327 MB。

**冻结产物（G 项）**：每个成员的加性项表 + booster 成分基与设计矩阵成分得分，逐文件 SHA256 见 `REPRODUCIBILITY_MANIFEST.json` 的 `artifact_checksums`；整套 `runs/final` 与我们的 `prediction.csv` 以 GitHub Release （tag `v2.0-semifinal` 的 Assets）提供稳定下载链接。本模型没有神经网络意义上的权重文件——训练在 CPU 上从头复现只需上表时间。

---

## 2. 模型

预测按评分口径拆成两部分：

```
y_hat(样本) = B_hat(批次结构)  +  Delta_hat(扰动效应)
```

**第一层 `UnifiedBackfit`（`src/vcell/models.py`）**——在 log2 空间对一列设计因子做**收缩回填**
（shrunken backfitting）。因子按「粗 → 细」排列，每个因子的每个水平对每个蛋白估一个偏移，
按 `n/(n+λ)` 向 0 收缩：

```
source(4) → instrument(7) → plate(144) → plate×strain(381) → strain(5)
          → strain×{medium, temp, time, source}
          → compound(56) → compound×{time, temp, medium, source, strain}
```

三个设计要点，每一个都由实测决定，不是默认值：

- **批次项与扰动项一起拟合**，而不是先用对照孔定基线。板效应单独解释 88% 的方差，
  只用对照孔会浪费 87% 的标签。总分 0.407 → 0.457。
- **`source` 与 `instrument` 必须排在 `plate` 之前**。两者完美嵌套于 plate 内（144 个板各只有
  1 台仪器），把它们挪到 plate 之后，增益从 +0.0056 塌到 +0.0013——这是**分层部分池化**的
  判定性签名：嵌套去掉的是"信息"，不是"部分池化"。
- **收缩强度 λ 是被重调过的**，不是初值。`plate` 0.3→2.0、`plate×strain` 2.0→6.0 值 +0.0056；
  扰动族整体 ×4 值 +0.0029。这两笔加起来比本项目所有"新模块"的总和还多一倍。

**第二层 `ResidualBooster`**——把 4,422 维残差 SVD 压到 240 个成分，每个成分一棵 LightGBM
森林（1600 树 / lr 0.015 / 3 个随机种子平均）。加性模型只能表达一阶和二阶表格，
高阶交互交给树。它值 +0.0105，且主要落在加性模型对未见化合物**完全无法移动**的 M3/M4 上。

**集成**——12 个成员 = 2 种因子拟合顺序 × 6 个收缩强度点，等权平均。见第 3 节。

---

## 3. 为什么是这 12 个成员

集成成员的价值不在于"各自更好"，而在于**偏差方向不同**。本项目的取舍全部由
**六个内层折的逐折配对检验**决定（`scripts/paired_fold_eval.py`），过线标准
`delta > 2 × sem`：

| 候选 | 单独用（vs A） | 作为集成成员（vs 基线集成） | 结论 |
|---|---|---|---|
| λ 邻域（plate 1/4、pert ×2/×8/×16） | −0.0020 ~ +0.0004 | 合计 +0.00097（6/6） | **采纳** |
| **strain 提前到 plate 之前**（F 族 6 个） | +0.0021，**仅 3/6，不过线** | **+0.0019（6/6）** | **采纳** |
| strain 族各 λ ×0.25/×4、`fit_offset` 关、`n_pass` 3/4/8、`plate×strain` 3/12 | 全在 ±0.0004 内 | −0.00015 ~ +0.00001，**全零** | 否决 |
| `pert ×32`、`booster 320 成分` | +0.0017 / +0.0009 | +0.00024（5/6）／ +0.00015 | **未采纳**（见下） |

第二行是本项目方法论上最重要的一条：**"作为替代方案被否决"的候选，恰恰是最好的集成成员**。
strain-early 顺序单独用时 3/6 折不过线，做成员却 6/6 过线，官方 val 镜像上它的六个变体
**每一个**都优于对应的当前顺序成员（0.5005–0.5036 vs 0.4970–0.4999）。反过来，所有
λ 邻域候选（低方差、与现有成员太像）对集成的贡献精确为零。

最后两行说明为什么停在 12：两个候选分别是 +0.00024（恰好等于 2×sem，仅 5/6）与
−0.00009，都没有过线，而每加一个成员就给复现多加约 30 分钟训练。**收益已经饱和。**
（附注：8/27 的一次评估曾给 Br 记 +0.00015；我们在 9/2 两次独立重算均为 −0.00009，
以可复现值为准，重算日志见 `artifacts/results/pool_real_eval.csv`。）

---

## 4. 数据边界与合规

| 项 | 做法 |
|---|---|
| 转导式设计 | 模型在 train+test **行**的并集上拟合设计矩阵（测试 metadata 官方公开、不含真值）；因此**训练时需要同时提供测试 metadata**。测试**标签**从未进入任何环节 |
| 训练标签 | **只有** `split_final == 'train'` 的 5,920 行。`configs/final.yaml` 里断言行数，不符即报错退出 |
| 验证集 | 只用于模型选择（`scripts/evaluate_val_mirror.py`，全程只评一次），不进入训练、不参与任何统计量 |
| 测试蛋白真值 | **代码层禁读**。官方数据包里的 `WAYB_WAYC_proteome_raw_test.csv` 含全部测试真值，已移入 `data/quarantine/`；`src/vcell/io.py::load_proteome` 收到 `'test'` 直接抛 `RuntimeError`。`scripts/build_embeddings.py --check` 会**实际调用一次**来证明它确实被拒 |
| 蛋白过滤 | 仅用 train 行计算缺失率，删除 ≥80% 缺失的蛋白 → 4,422 个。名称与顺序取自官方训练文件表头（含带逗号的 `ARG5,6` / `DUR1,2`） |
| 尺度 | 对有限且 >0 的 raw intensity 取 log2；NA 保留为 mask，**不填 0**。提交为 log2，`prediction_manifest.json` 声明 `prediction_scale` |
| 归一化统计 | 蛋白均值、SVD 基、类别词表全部只在 train 行上拟合 |
| 随机种子 | booster `seeds=[0,1,2]`，每个成分 j 的 `random_state = seed + j`；加性模型完全确定 |
| 外部数据 | **最终模型一处都不用**。探索过但未采用的资源（1,011 酵母基因组、SGD、PubChem）连同 URL、版本、下载日期、许可证与 SHA256 全部登记在 `external_data/source_manifest.json` |
| 许可证 | 本作品 MIT（`LICENSES/`）。依赖：numpy / pandas / scipy / LightGBM（均 BSD 或 MIT） |
| 商业 API / 闭源模型 | **未使用** |

### 数值可复现性

加性模型完全确定。LightGBM 在不同线程数下有约 1e-3 log2 量级的差异——实测同一配置
32 线程 vs 16 线程：逐元素 rms 差 0.003，相关 0.9999995，对任何指标的影响在小数点后第五位。
线程数只影响末位数值，默认设置即可复现到该精度；我们的正式产物在 16–32 线程混合下生成，上表的一致性核对（rms 0.0008）覆盖了这一差异。

---

## 5. 目录

```
README.md                     本文
requirements.txt              依赖与版本
configs/final.yaml            ⭐ 唯一配置真源：12 个成员、全部 λ、booster、断言
src/vcell/
  ├─ io.py                    数据载入 + 测试真值隔离守卫
  ├─ models.py                UnifiedBackfit / ResidualBooster（模型本体）
  ├─ pipeline.py              训练与推理的分离层：拟合 / 冻结 / 重建
  ├─ harness.py               折构造、蛋白过滤、评估入口
  ├─ metrics.py               六个官方评分模块的忠实复现
  └─ design.py                对照匹配
scripts/
  ├─ build_embeddings.py      主命令 1（自证无外部数据）
  ├─ train.py                 主命令 2（从头训练 12 个成员）
  ├─ predict.py               主命令 3（冻结推理 → prediction.csv）
  ├─ validate_submission.py   提交格式自检
  ├─ smoke_test.sh            3 分钟冒烟
  ├─ evaluate_val_mirror.py   官方 val 镜像评估（模型选择用，只评一次）
  ├─ paired_fold_eval.py      ⭐ 六折逐折配对检验（所有取舍的裁判）
  └─ experiments/             全部实验脚本，含被否决的方向
external_data/source_manifest.json   外部资源申报（含 SHA256）
artifacts/results/            逐折原始结果与台账（评审据此核对取舍）
tests/                        单元测试
```

---

## 6. 已知限制

1. **未见菌株的基线不可估**。留出菌株的每蛋白基线偏移 b（rms 0.34–0.36 log2，比真实扰动效应
   0.146 大一倍多）在评分里不对称：真值那边被对照减掉了，预测这边没有。我们量化过它的
   外部数据上界——公开基因组只能标记 2% 的（菌株, 蛋白）对、覆盖 b 方差的 7.7%，完美修正
   后 rms 仅从 0.428 降到 0.411，换算到总分约 +0.0003（`scripts/experiments/63-65`）。
   这是本作品最大的、且**已被定量封边**的限制。
2. **见过菌株的划分已贴到噪声地板**。chem_only / time 的随机残差 0.306 / 0.301，
   而单样本测量噪声地板是 0.26，比值 1.18 / 1.16。那里没有多少可提的了。
3. **评分口径的不确定性大于建模改进**。同一份预测在不同合理读法下总分跨度 0.410–0.574，
   是我们全部提分幅度的十几倍。相关问题已整理成清单提给组委会。
4. **集成成本**。12 个成员意味着复现需要约 6 小时 CPU；`--members` 支持分批并行。
5. **DHY210 无公开基因组**。它是实验室菌株，不在 1,011 基因组集合内；诊断实验中按 S288c
   参考处理。该假设**不影响最终模型**（最终模型不用基因组）。

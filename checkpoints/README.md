# checkpoints/

本模型**没有神经网络意义上的权重文件**。训练产物（G 项的"最终 checkpoint"）是
`scripts/train.py` 写出的 `runs/final/`（约 327 MB）：每个集成成员的加性项表、
booster 成分基 V 与设计矩阵成分得分，逐文件 SHA256 已登记在
`REPRODUCIBILITY_MANIFEST.json` 的 `artifact_checksums`。

获取方式（二选一）：
1. **从头复现**（推荐，官方验收流程即此）：README 的主命令 2，约 6 小时 CPU；
2. **稳定下载**：GitHub Release（仓库 tag `v2.2-semifinal` 的 Assets）提供
   `runs_final.tar.gz` 与我们的 `prediction.csv`，SHA256 与 manifest 一致。

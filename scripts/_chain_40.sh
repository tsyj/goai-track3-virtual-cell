#!/bin/bash
# 39 结束后接着跑 40 (把细粒度扰动身份也给 booster)
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! LC_ALL=C grep -aq "^total " results/39_variant.log 2>/dev/null; do
    sleep 60; waited=$((waited+60))
    [ $waited -gt 14400 ] && { echo "等 39 超过 4 小时, 中止"; exit 3; }
done
echo "39 完成 (等待 ${waited}s), 开始 40"
VCELL_WORKERS=14 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/40_booster_variant_feature.py > results/40_varfeat.log 2>&1
echo "40 结束"
sed -n '/=== mean over/,$p' results/40_varfeat.log

#!/bin/bash
# 40 结束后接着跑 41 (钉死 instrument 层级)
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! LC_ALL=C grep -aq "^total " results/40_varfeat.log 2>/dev/null; do
    sleep 60; waited=$((waited+60))
    [ $waited -gt 10800 ] && { echo "等 40 超过 3 小时, 中止"; exit 3; }
done
echo "40 完成 (等待 ${waited}s), 开始 41"
VCELL_WORKERS=14 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/41_instrument_level.py > results/41_instr.log 2>&1
echo "41 结束"
sed -n '/=== with cheap booster/,$p' results/41_instr.log

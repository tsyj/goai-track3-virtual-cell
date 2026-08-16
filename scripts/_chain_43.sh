#!/bin/bash
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! LC_ALL=C grep -aq "^total " results/42_confirm.log 2>/dev/null; do
    sleep 60; waited=$((waited+60))
    [ $waited -gt 10800 ] && { echo "等 42 超过 3 小时, 中止"; exit 3; }
done
echo "42 完成 (等待 ${waited}s), 开始 43"
VCELL_WORKERS=14 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/43_condition_main_effects.py > results/43_cond.log 2>&1
echo "43 结束"
sed -n '/=== with cheap booster/,$p' results/43_cond.log | head -12

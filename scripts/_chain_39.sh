#!/bin/bash
# 38 结束后接着跑 39 (扰动身份粒度 + instrument 层级)
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
# 先等 38 起来再等它结束; 若 38 迟迟不启动也不要空转
while ! LC_ALL=C grep -aq "^total " results/38_more.log 2>/dev/null; do
    sleep 60; waited=$((waited+60))
    [ $waited -gt 10800 ] && { echo "等 38 超过 3 小时, 中止"; exit 3; }
done
echo "38 完成 (等待 ${waited}s), 开始 39"
VCELL_WORKERS=14 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/39_pert_variant.py > results/39_variant.log 2>&1
echo "39 结束"
sed -n '/=== with cheap booster/,$p' results/39_variant.log

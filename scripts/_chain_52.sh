#!/bin/bash
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! LC_ALL=C grep -aq "^total " results/10_predict_v4.log 2>/dev/null; do
    pgrep -f "10_predict_test|14_final_eval" >/dev/null || { echo "定稿跑批消失"; break; }
    sleep 60; waited=$((waited+60))
    [ $waited -gt 7200 ] && break
done
echo "定稿跑批完成 (等待 ${waited}s)"
LC_ALL=C grep -a "R6\|wrote\|^total" results/14_final_v4.log results/10_predict_v4.log 2>/dev/null | head -6
echo "=== 开始 52 (booster 正则化族) ==="
VCELL_WORKERS=14 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/52_booster_regularisation.py > results/52_reg.log 2>&1
echo "52 结束"
sed -n '/=== six orphan-free/,$p' results/52_reg.log

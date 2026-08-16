#!/bin/bash
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! LC_ALL=C grep -aq "^total " results/10_predict_v3.log 2>/dev/null; do
    pgrep -f "10_predict_test|14_final_eval" >/dev/null || { echo "最终跑批消失且无 total"; break; }
    sleep 60; waited=$((waited+60))
    [ $waited -gt 7200 ] && { echo "等最终跑批超时"; exit 3; }
done
echo "最终跑批完成 (等待 ${waited}s), 开始 44"
VCELL_WORKERS=14 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/44_retune_plate_lambda.py > results/44_retune.log 2>&1
echo "44 结束"
/home/xinyuan/anaconda3/envs/numpy1/bin/python scripts/analyze_paired.py \
  results/retune_plate_raw.csv "current (plate .3, pxs 2)" 2>&1 | tail -18

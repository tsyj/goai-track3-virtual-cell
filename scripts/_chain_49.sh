#!/bin/bash
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! LC_ALL=C grep -aq "^total " results/48_pert.log 2>/dev/null; do
    pgrep -f "48_pert_lambda_wide" >/dev/null || { echo "48 进程消失且无 total"; break; }
    sleep 60; waited=$((waited+60))
    [ $waited -gt 7200 ] && { echo "等 48 超时"; exit 3; }
done
echo "48 完成 (等待 ${waited}s)"
/home/xinyuan/anaconda3/envs/numpy1/bin/python scripts/analyze_paired.py \
  results/family_lambda2_raw.csv "strain x1, pert x4" 2>&1 | head -10
echo "=== 开始 49 ==="
VCELL_WORKERS=14 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/49_factor_order.py > results/49_order.log 2>&1
echo "49 结束"
/home/xinyuan/anaconda3/envs/numpy1/bin/python scripts/analyze_paired.py \
  results/factor_order_raw.csv "current order, n_pass 6" 2>&1 | head -12

#!/bin/bash
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! LC_ALL=C grep -aq "^total " results/46_confirm_lam.log 2>/dev/null; do
    pgrep -f "46_confirm_plate_lam" >/dev/null || { echo "46 进程消失且无 total"; break; }
    sleep 60; waited=$((waited+60))
    [ $waited -gt 7200 ] && { echo "等 46 超时"; exit 3; }
done
echo "46 完成 (等待 ${waited}s)"
/home/xinyuan/anaconda3/envs/numpy1/bin/python scripts/analyze_paired.py \
  results/confirm_plate_lam_raw.csv "incumbent (plate .3, pxs 2)" 2>&1 | tail -6
echo "=== 开始 47 ==="
VCELL_WORKERS=14 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/47_family_lambdas.py > results/47_family.log 2>&1
echo "47 结束"
/home/xinyuan/anaconda3/envs/numpy1/bin/python scripts/analyze_paired.py \
  results/family_lambda_raw.csv "strain x1, pert x1" 2>&1 | tail -14

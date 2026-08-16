#!/bin/bash
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! LC_ALL=C grep -aq "^total " results/50_final.log 2>/dev/null; do
    pgrep -f "50_confirm_final" >/dev/null || { echo "50 进程消失且无 total"; break; }
    sleep 60; waited=$((waited+60))
    [ $waited -gt 7200 ] && { echo "等 50 超时"; exit 3; }
done
echo "50 完成 (等待 ${waited}s)"
/home/xinyuan/anaconda3/envs/numpy1/bin/python scripts/analyze_paired.py \
  results/confirm_final_raw.csv "adopted (pert x1)" 2>&1 | tail -5
echo "=== 开始 51 (六个无孤儿折) ==="
VCELL_WORKERS=12 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/51_orphan_free_folds.py > results/51_free.log 2>&1
echo "51 结束"
sed -n '/=== six orphan-free/,$p' results/51_free.log

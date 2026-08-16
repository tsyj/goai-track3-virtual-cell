#!/bin/bash
# 等最终评估与提交生成收尾, 再跑 38 (把 bagging / 成分数 / 步长 再推一档), 避免抢核
cd /home/xinyuan/比赛/AIGO赛道三算法题
waited=0
while ! (LC_ALL=C grep -aq "^total " results/14_final_v2.log 2>/dev/null \
         && LC_ALL=C grep -aq "^total " results/10_predict_v2.log 2>/dev/null); do
    pgrep -f "14_final_eval|10_predict_test" > /dev/null || {
        echo "14/10 进程消失且无 total 行, 中止链"; exit 2; }
    sleep 60; waited=$((waited+60))
    [ $waited -gt 7200 ] && { echo "等待超过 2 小时, 中止"; exit 3; }
done
echo "14/10 完成 (等待 ${waited}s), 开始 38"
VCELL_WORKERS=12 VCELL_LGB_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 /home/xinyuan/anaconda3/envs/numpy1/bin/python -u \
  scripts/38_booster_more.py > results/38_more.log 2>&1
echo "38 结束"
sed -n '/=== mean over/,$p' results/38_more.log

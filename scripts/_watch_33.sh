#!/bin/bash
# 33_booster_knobs 健康检查: 日志 / 结果 CSV / CPU 三信号联合, 任一有进度即算活
cd /home/xinyuan/比赛/AIGO赛道三算法题
LOG=results/33_knobs.log
CSV=results/booster_knobs_raw.csv
prev_log=0; prev_csv=0; stall=0
while true; do
    sleep 180
    cur_log=$(stat -c%s $LOG 2>/dev/null || echo 0)
    cur_csv=$(stat -c%s $CSV 2>/dev/null || echo 0)
    cpu=$(ps -eo pcpu,args | LC_ALL=C grep -a "33_booster_knobs" | grep -v grep \
          | awk '{s+=$1} END{printf "%d", s+0}')

    if LC_ALL=C grep -aq "^total " $LOG 2>/dev/null; then
        echo "KNOBS_DONE"; tail -18 $LOG; exit 0
    fi
    if ! pgrep -f "33_booster_knobs" > /dev/null; then
        echo "KNOBS_DIED (无 total 行, 进程已消失)"; tail -20 $LOG; exit 2
    fi
    if [ "$cur_log" -gt "$prev_log" ] || [ "$cur_csv" -gt "$prev_csv" ] || [ "${cpu:-0}" -gt 50 ]; then
        stall=0
    else
        stall=$((stall+1))
        if [ $stall -ge 5 ]; then
            echo "KNOBS_STALL: 日志/CSV/CPU 三信号 15 分钟全静止 (log=${cur_log}B csv=${cur_csv}B cpu=${cpu}%)"
            tail -8 $LOG; exit 1
        fi
    fi
    prev_log=$cur_log; prev_csv=$cur_csv
done

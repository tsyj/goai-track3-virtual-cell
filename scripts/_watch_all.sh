#!/bin/bash
# 常驻健康检查: 三信号联合(日志 / 输出文件 / CPU), 任一有进度即算活; 全静止 15 分钟才告警。
# 覆盖 14_final_eval / 10_predict_test / 38_booster_more / 39_* 等后续跑批。
cd /home/xinyuan/比赛/AIGO赛道三算法题
PAT="numpy1/bin/python -u scripts/|_chain_3|_chain_4"
declare -A prev
stall=0
while true; do
    sleep 180
    running=$(pgrep -fc "$PAT")
    sig=0
    for f in results/14_final_v2.log results/10_predict_v2.log results/38_more.log \
             results/39_variant.log results/40_varfeat.log results/pert_variant_raw.csv results/variant_feature_raw.csv submission/prediction.csv results/booster_more_raw.csv; do
        [ -e "$f" ] || continue
        cur="$(stat -c%s%Y "$f" 2>/dev/null)"
        [ "${prev[$f]}" != "$cur" ] && sig=1
        prev[$f]="$cur"
    done
    cpu=$(ps -eo pcpu,args | LC_ALL=C grep -aE "$PAT" | grep -v grep | awk '{s+=$1} END{printf "%d", s+0}')
    [ "${cpu:-0}" -gt 50 ] && sig=1

    if [ "$running" -eq 0 ]; then
        echo "IDLE: 没有跑批在运行 (14/10/38 都已结束或未启动)"
        for f in results/14_final_v2.log results/10_predict_v2.log results/38_more.log; do
            [ -e "$f" ] && echo "  $f: $(LC_ALL=C grep -ac '^total ' "$f" 2>/dev/null) 个 total 行"
        done
        exit 0
    fi
    if [ $sig -eq 1 ]; then
        stall=0
    else
        stall=$((stall+1))
        if [ $stall -ge 5 ]; then
            echo "STALL: 日志/输出/CPU 三信号 15 分钟全静止, 但仍有 $running 个进程 (cpu=${cpu}%)"
            for f in results/14_final_v2.log results/10_predict_v2.log results/38_more.log; do
                [ -e "$f" ] && { echo "--- $f"; LC_ALL=C grep -av "UserWarning\|_log_warning" "$f" | tail -3; }
            done
            exit 1
        fi
    fi
done

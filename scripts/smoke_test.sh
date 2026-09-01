#!/usr/bin/env bash
# 3 分钟冒烟：用 1 个成员 + 缩减 booster 跑通 训练 → 推理 → 校验 全链路。
# 只验证环境与代码可用，不产生正式结果（正式结果见 README 的三条主命令）。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$(command -v python3 || command -v python)}"

echo "=== 0/4 环境 ==="
$PY -c "import numpy,pandas,scipy,lightgbm,yaml;print('numpy',numpy.__version__,'pandas',pandas.__version__,'lightgbm',lightgbm.__version__)"

echo "=== 1/4 外部数据自证 ==="
$PY scripts/build_embeddings.py --check

echo "=== 2/4 训练（1 个成员，缩减 booster）==="
tmp=$(mktemp -d)
$PY - "$tmp/smoke.yaml" <<'PYEOF'
import sys, yaml
c = yaml.safe_load(open("configs/final.yaml", encoding="utf-8"))
c["model"]["booster"] = {"n_comp": 16, "n_estimators": 40, "learning_rate": 0.1,
                         "num_leaves": 31, "seeds": [0]}
c["model"]["n_pass"] = 2
c["ensemble"]["members"] = [m for m in c["ensemble"]["members"] if m["name"] == "A"]
yaml.safe_dump(c, open(sys.argv[1], "w", encoding="utf-8"), allow_unicode=True)
print("写出", sys.argv[1])
PYEOF
$PY scripts/train.py --config "$tmp/smoke.yaml" --output-dir "$tmp/run"

echo "=== 3/4 推理 ==="
$PY scripts/predict.py --config "$tmp/smoke.yaml" --run-dir "$tmp/run" \
    --output "$tmp/prediction_smoke.csv"

echo "=== 4/4 格式校验 ==="
$PY scripts/validate_submission.py --prediction "$tmp/prediction_smoke.csv"

echo
echo "冒烟通过。临时文件在 $tmp（可删）。"
echo "注意：这是缩减配置，分数不代表最终模型；正式结果请按 README 的三条主命令运行。"

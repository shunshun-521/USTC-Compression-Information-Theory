#!/usr/bin/env bash
# 运行全部极化码实验（可通过环境变量加速）
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
export POLAR_MAX_FRAMES="${POLAR_MAX_FRAMES:-5000}"
export POLAR_MIN_ERRORS="${POLAR_MIN_ERRORS:-30}"

echo "=== 极化码实验开始 (MAX_FRAMES=$POLAR_MAX_FRAMES, MIN_ERRORS=$POLAR_MIN_ERRORS) ==="
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
echo "=== 全部实验完成 ==="

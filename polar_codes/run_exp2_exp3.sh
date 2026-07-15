#!/usr/bin/env bash
# 运行实验二、三（实验一已完成时可单独调用）
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
export POLAR_MAX_FRAMES="${POLAR_MAX_FRAMES:-2000}"
export POLAR_MIN_ERRORS="${POLAR_MIN_ERRORS:-20}"

echo "=== 实验二、三 (MAX_FRAMES=$POLAR_MAX_FRAMES) ==="
python3 run_exp2.py
python3 run_exp3.py
echo "=== 完成 ==="

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export POLAR_MAX_FRAMES="${POLAR_MAX_FRAMES:-50000}"
export POLAR_MIN_ERRORS="${POLAR_MIN_ERRORS:-100}"
export POLAR_EB_STEP="${POLAR_EB_STEP:-0.25}"

echo "=== 开始完整实验 ==="
echo "POLAR_MAX_FRAMES=$POLAR_MAX_FRAMES POLAR_MIN_ERRORS=$POLAR_MIN_ERRORS"
echo "顺序: exp3 -> exp2 -> regenerate_plots"
date -u '+%Y-%m-%d %H:%M:%S UTC'

echo ""
echo ">>> 实验三 (run_exp3.py)"
python3 -u run_exp3.py 2>&1 | tee results/exp3_full.log

echo ""
echo ">>> 实验二 (run_exp2.py)"
python3 -u run_exp2.py 2>&1 | tee results/exp2_full.log

echo ""
echo ">>> 重绘图表 (regenerate_plots.py)"
python3 -u regenerate_plots.py 2>&1 | tee -a results/exp3_full.log

echo ""
echo "=== 全部完成 ==="
date -u '+%Y-%m-%d %H:%M:%S UTC'

#!/bin/bash
# 顺序运行全部极化码实验（建议在后台执行）
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p results

echo "===== 实验一：SC 仿真 ====="
python3 run_exp1.py 2>&1 | tee results/exp1.log

echo "===== 实验二：SCL 仿真 ====="
python3 run_exp2.py 2>&1 | tee results/exp2.log

echo "===== 实验三：BP 仿真 ====="
python3 run_exp3.py 2>&1 | tee results/exp3.log

echo "===== 全部实验完成 ====="
ls -la results/

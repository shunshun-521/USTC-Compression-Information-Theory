#!/bin/bash
# 顺序运行全部实验（可通过环境变量调整仿真规模）
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
echo "全部实验完成。"

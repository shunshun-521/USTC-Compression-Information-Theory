#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== 实验一 SC ==="
python3 run_exp1.py
echo "=== 实验二 SCL ==="
python3 run_exp2.py
echo "=== 实验三 BP ==="
python3 run_exp3.py
echo "=== 全部完成 ==="

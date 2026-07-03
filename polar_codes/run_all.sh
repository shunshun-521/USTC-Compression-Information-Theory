#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== 实验一 ===" && python3 run_exp1.py
echo "=== 实验二 ===" && python3 run_exp2.py
echo "=== 实验三 ===" && python3 run_exp3.py
echo "=== 全部完成 ==="

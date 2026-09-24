#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== 极化码仿真验证 ==="
python3 verify.py
echo ""
echo "=== 实验一: SC 仿真 ==="
python3 run_exp1.py
echo ""
echo "=== 实验二: SCL 仿真 ==="
python3 run_exp2.py
echo ""
echo "=== 实验三: BP 仿真 ==="
python3 run_exp3.py
echo ""
echo "=== 全部完成 ==="
ls -la results/

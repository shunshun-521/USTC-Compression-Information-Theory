#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 verify.py
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py

echo "全部实验完成，结果见 results/ 目录。"

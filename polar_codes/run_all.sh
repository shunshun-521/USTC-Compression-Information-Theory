#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
python3 verify.py
python3 run_exp1.py 2>&1 | tee results/exp1_run.log
python3 run_exp2.py 2>&1 | tee results/exp2_run.log
python3 run_exp3.py 2>&1 | tee results/exp3_run.log
echo "All experiments completed."

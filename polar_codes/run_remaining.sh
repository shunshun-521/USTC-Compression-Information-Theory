#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
python3 run_exp2.py
python3 run_exp3.py
echo ALL_EXPERIMENTS_DONE

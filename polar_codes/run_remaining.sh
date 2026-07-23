#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 run_exp2.py
python3 run_exp3.py
echo "All experiments completed."

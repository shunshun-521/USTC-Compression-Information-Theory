#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Running polar code experiments..."
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
echo "All experiments completed."

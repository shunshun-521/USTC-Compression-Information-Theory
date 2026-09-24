#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Running verification..."
python3 verify.py
echo "Running experiment 1 (SC)..."
python3 run_exp1.py
echo "Running experiment 2 (SCL)..."
python3 run_exp2.py
echo "Running experiment 3 (BP)..."
python3 run_exp3.py
echo "All experiments complete."

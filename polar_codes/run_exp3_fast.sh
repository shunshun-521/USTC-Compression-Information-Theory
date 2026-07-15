#!/usr/bin/env bash
# 快速完成实验三
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export POLAR_MAX_FRAMES=1500
export POLAR_MIN_ERRORS=15
python3 run_exp3.py

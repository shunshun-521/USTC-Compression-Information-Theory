#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 verify.py
if [[ "${POLAR_QUICK:-}" == "1" ]]; then
  export POLAR_QUICK=1
  export POLAR_MAX_FRAMES=2000
  export POLAR_MIN_ERRORS=20
fi
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py

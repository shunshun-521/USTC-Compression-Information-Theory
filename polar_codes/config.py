"""仿真规模配置（可通过环境变量覆盖，便于自动化快速跑通）。"""
import os
import numpy as np

POLAR_QUICK = os.environ.get("POLAR_QUICK", "0") == "1"
POLAR_MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "5000" if POLAR_QUICK else "100000"))
POLAR_MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "20" if POLAR_QUICK else "100"))
POLAR_SKIP_N1024 = os.environ.get("POLAR_SKIP_N1024", "1" if POLAR_QUICK else "0") == "1"
POLAR_N256_ONLY = os.environ.get("POLAR_N256_ONLY", "0") == "1"

if POLAR_QUICK:
    POLAR_EB_N0_RANGE_EXP1 = np.arange(1.0, 4.0, 0.5)
    POLAR_EB_N0_RANGE_EXP23 = np.arange(1.5, 4.5, 0.5)
else:
    POLAR_EB_N0_RANGE_EXP1 = np.arange(0.0, 5.5, 0.25)
    POLAR_EB_N0_RANGE_EXP23 = np.arange(1.0, 5.5, 0.25)

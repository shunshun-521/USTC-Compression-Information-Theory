"""仿真规模配置（环境变量 POLAR_QUICK=1 可快速冒烟）"""
import os

QUICK = os.getenv("POLAR_QUICK", "0") == "1"

if QUICK:
    MAX_FRAMES = 800
    MIN_ERRORS = 15
else:
    MAX_FRAMES = 100000
    MIN_ERRORS = 100

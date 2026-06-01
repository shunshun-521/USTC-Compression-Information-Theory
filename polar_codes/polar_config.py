"""仿真参数：可通过环境变量 POLAR_QUICK=1 缩短蒙特卡洛时间。"""
import os

QUICK = os.environ.get("POLAR_QUICK", "").strip() in ("1", "true", "yes")

MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "5000" if QUICK else "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "50" if QUICK else "100"))

"""实验参数配置，支持环境变量覆盖。"""
import os

_QUICK = os.environ.get("POLAR_QUICK", "0") == "1"

MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "5000" if _QUICK else "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "20" if _QUICK else "100"))

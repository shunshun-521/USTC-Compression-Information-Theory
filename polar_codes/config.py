"""实验参数（可通过环境变量覆盖，便于快速冒烟测试）。"""
import os

import numpy as np


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


MAX_FRAMES = env_int("POLAR_MAX_FRAMES", 100000)
MIN_ERRORS = env_int("POLAR_MIN_ERRORS", 100)
DESIGN_EBN0 = env_float("POLAR_DESIGN_EBN0", 2.5)
EB_N0_MIN = env_float("POLAR_EB_MIN", 0.0)
EB_N0_MAX = env_float("POLAR_EB_MAX", 5.5)
EB_N0_STEP = env_float("POLAR_EB_STEP", 0.25)

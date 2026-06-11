"""实验参数配置，支持环境变量覆盖。"""
import os

_QUICK = os.environ.get("POLAR_QUICK", "0") == "1"

MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "2000" if _QUICK else "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "20" if _QUICK else "100"))

# 快速模式下缩减码长与信噪比扫描范围以控制运行时间
EXP1_N_LIST = (
    [256, 512]
    if _QUICK and "POLAR_EXP1_N_LIST" not in os.environ
    else [256, 512, 1024]
)
EXP3_N_LIST = (
    [256]
    if _QUICK and "POLAR_EXP3_N_LIST" not in os.environ
    else [256, 512]
)

"""实验脚本共用环境变量配置（便于 CI / 云代理快速跑通）"""
import os
import numpy as np


def env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default


def apply_sim_limits(max_frames: int, min_errors: int):
    max_frames = env_int("POLAR_MAX_FRAMES", max_frames)
    min_errors = env_int("POLAR_MIN_ERRORS", min_errors)
    if os.environ.get("POLAR_QUICK") == "1":
        max_frames = min(max_frames, 2000)
        min_errors = min(min_errors, 20)
    return max_frames, min_errors


def filter_n_list(n_list):
    if os.environ.get("POLAR_N256_ONLY") == "1":
        return [n for n in n_list if n == 256]
    if os.environ.get("POLAR_SKIP_N1024") == "1":
        return [n for n in n_list if n != 1024]
    return n_list


def eb_n0_range(default):
    raw = os.environ.get("POLAR_EB_N0_RANGE")
    if not raw:
        if os.environ.get("POLAR_QUICK") == "1":
            return np.arange(2.0, 3.5, 0.5)
        return default
    return np.array([float(x) for x in raw.split(",")])

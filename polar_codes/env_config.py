"""读取环境变量以控制仿真规模（Cloud/CI 可缩短运行时间）"""
import os
import numpy as np


def _int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def sim_config(default_max_frames=100000, default_min_errors=100):
    max_frames = _int("POLAR_MAX_FRAMES", default_max_frames)
    min_errors = _int("POLAR_MIN_ERRORS", default_min_errors)
    eb_min = _float("POLAR_EB_MIN", None)
    eb_max = _float("POLAR_EB_MAX", None)
    eb_step = _float("POLAR_EB_STEP", None)
    return max_frames, min_errors, eb_min, eb_max, eb_step


def eb_n0_range(default):
    _, _, eb_min, eb_max, eb_step = sim_config()
    if eb_min is not None and eb_max is not None and eb_step is not None:
        return np.arange(eb_min, eb_max + 1e-9, eb_step)
    return default


def parse_n_list(default_list):
    raw = os.environ.get("POLAR_N_LIST")
    if not raw:
        return default_list
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_l_list(default_list):
    raw = os.environ.get("POLAR_L_LIST")
    if not raw:
        return default_list
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def skip_n1024(default_list):
    if os.environ.get("POLAR_SKIP_N1024", "").lower() in ("1", "true", "yes"):
        return [n for n in default_list if n != 1024]
    return default_list

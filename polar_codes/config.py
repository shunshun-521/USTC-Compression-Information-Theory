"""Optional environment overrides for faster smoke tests."""
import os
import numpy as np


def _env_int(name, default):
    val = os.environ.get(name)
    return int(val) if val is not None else default


def _env_float(name, default):
    val = os.environ.get(name)
    return float(val) if val is not None else default


MAX_FRAMES = _env_int("POLAR_MAX_FRAMES", 100_000)
MIN_ERRORS = _env_int("POLAR_MIN_ERRORS", 100)
BP_MAX_ITER = _env_int("POLAR_BP_MAX_ITER", 50)

EB_N0_MIN = _env_float("POLAR_EB_MIN", None)
EB_N0_MAX = _env_float("POLAR_EB_MAX", None)
EB_N0_STEP = _env_float("POLAR_EB_STEP", None)


def n_list_exp1(default):
    val = os.environ.get("POLAR_N_LIST")
    if val:
        return [int(x) for x in val.split(",")]
    return default


def eb_n0_range(default):
    if EB_N0_MIN is not None and EB_N0_MAX is not None and EB_N0_STEP is not None:
        return np.arange(EB_N0_MIN, EB_N0_MAX + 1e-9, EB_N0_STEP)
    return default

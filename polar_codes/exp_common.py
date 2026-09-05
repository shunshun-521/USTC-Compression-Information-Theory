"""实验脚本公共：单元测试与快速仿真参数"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def run_unit_tests():
    from validate import (
        test_construction,
        test_crc,
        test_encoder,
        test_sc_lossless,
        test_sc_recursive_match,
        test_scl_l1_equals_sc,
    )

    test_encoder()
    test_crc()
    test_construction()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("单元测试全部通过。\n")


def sim_params(default_max=100000, default_min=100, default_eb=None):
    """读取环境变量以支持快速仿真"""
    fast = os.environ.get("POLAR_FAST_SIM", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", 500 if fast else default_max))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", 20 if fast else default_min))
    if default_eb is not None:
        eb = default_eb
        if fast:
            eb = default_eb[::2] if len(default_eb) > 3 else default_eb
        eb = np.asarray(eb, dtype=float)
    else:
        eb = None
    return max_frames, min_errors, eb

"""极化码模块公共校验与快速仿真参数。"""
import os

import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    """运行编译码模块单元测试。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    sc_errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0.0, sigma, size=N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在 Eb/N0=10dB 失败 {sc_errors}/100 帧"

    u_test = np.zeros(N, dtype=np.int8)
    u_test[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
    x_test = polar_encode(u_test)
    llr_test = compute_llr(bpsk_modulate(x_test), 0.01)
    uh_sc = sc_decode(llr_test, frozen_bits)
    uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(uh_sc, uh_scl), "L=1 的 SCL 应与 SC 等价"


def quick_params(default_max_frames, default_min_errors, default_eb_range):
    """根据环境变量返回仿真规模。"""
    if os.environ.get("POLAR_QUICK", "0") == "1":
        max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "2000"))
        min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "20"))
        eb_range = default_eb_range[::2] if len(default_eb_range) > 4 else default_eb_range
        return max_frames, min_errors, eb_range
    return default_max_frames, default_min_errors, default_eb_range

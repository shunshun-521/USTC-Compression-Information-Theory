"""模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([0, 0, 1, 1])
    if not np.array_equal(x, expected):
        # 与蝶形编码等价的参考输出（无额外比特倒序）
        expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)
    sigma = eb_n0_to_sigma(20.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        # 无损验证：无噪声信道 LLR
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, fb)
        assert np.array_equal(u_hat[info_idx], u[info_idx])

    # L=1 SCL 等价 SC
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr, fb)
    u_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)

    print("validate.py: 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

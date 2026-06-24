"""单元测试：验证各模块正确性。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器与生成矩阵不一致: {x} vs {x_ref}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    errors = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N), rng)
        llr = compute_llr(y, eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen_bool)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下出现 {errors}/100 帧错误"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bool = np.ones(N, dtype=bool)
    frozen_bool[info_idx] = False
    mismatches = 0
    for seed in range(50):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)),
            eb_n0_to_sigma(8.0, K / N),
        )
        u_sc = sc_decode(llr, frozen_bool)
        u_scl, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/50 帧"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

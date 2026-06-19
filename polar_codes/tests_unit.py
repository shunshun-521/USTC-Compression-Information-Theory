"""
极化码模块单元测试（各实验脚本运行前调用）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import encode_via_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = encode_via_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 无损译码失败"


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(456)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        errors += int(not np.array_equal(u_hat[info_idx], u[info_idx]))
    assert errors == 0, f"高信噪比 SC 译码错误帧数: {errors}"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(789)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-4)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 时 SCL 应等价于 SC"


def run_unit_tests():
    test_encoder()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

"""
极化码模块单元测试
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import build_generator_matrix, polar_encode


def test_encoder_matrix_consistency():
    """蝶形编码应与矩阵乘法一致。"""
    for N in (4, 8, 16, 64):
        G = build_generator_matrix(N)
        for _ in range(20):
            u = np.random.randint(0, 2, N)
            x_bf = polar_encode(u)
            x_mat = np.mod(u @ G, 2)
            assert np.array_equal(x_bf, x_mat), f"N={N}: {x_bf} vs {x_mat}"


def test_encoder_example():
    """N=4 编码示例（与 G_N 矩阵乘法一致）。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    expected = np.mod(u @ G, 2)
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


def test_sc_noiseless():
    """极低噪声下 SC 译码应完全正确。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u)


def test_sc_recursive_vs_nonrecursive():
    """递归与非递归 SC 结果一致。"""
    N = 32
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.random.default_rng(1).normal(0, 2, N)
    u1 = sc_decode_recursive(llr, frozen_bits.astype(bool))
    u2 = sc_decode(llr, frozen_bits)
    assert np.array_equal(u1, u2)


def test_scl_l1_equals_sc():
    """单路径 SCL 应等价于 SC。"""
    N = 64
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(5.0, 0.5)

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, N // 2)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = crc_encode(np.array([1, 0, 1, 1, 0, 0, 1, 0]), 8)
    assert crc_check(bits, 8)
    bad = bits.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8)


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8


def run_all():
    test_encoder_matrix_consistency()
    test_encoder_example()
    test_ga_construction()
    test_sc_recursive_vs_nonrecursive()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

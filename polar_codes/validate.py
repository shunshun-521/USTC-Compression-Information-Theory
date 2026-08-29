"""单元测试与数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("✓ 编码器校验通过")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    expected_info8 = np.array([0, 3, 5, 6])
    assert np.array_equal(info8, expected_info8), f"N=8 info错误: {info8}"
    print("✓ GA 构造 N=8 校验通过")

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected_first20 = np.array([
        1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22,
        25, 26, 28, 31, 32, 35, 37, 38,
    ])
    assert np.array_equal(info256[:20], expected_first20), (
        f"N=256 first20错误: {info256[:20]}"
    )
    print("✓ GA 构造 N=256 校验通过")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(15.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, size=K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("✓ SC 无损译码校验通过 (N=64, Eb/N0=10dB)")


def test_sc_recursive_matches():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(3.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, size=K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性校验通过")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(2.5, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, size=K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL L=1 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded_bad = coded.copy()
    coded_bad[0] ^= 1
    assert not crc_check(coded_bad, 8)
    print("✓ CRC 校验通过")


def test_f_g_operations():
    La, Lb = 2.5, -1.3
    f = f_operation(La, Lb)
    g0 = g_operation(La, Lb, 0)
    g1 = g_operation(La, Lb, 1)
    assert abs(f - (-1.3)) < 1e-10
    assert abs(g0 - (La + Lb)) < 1e-10
    assert abs(g1 - (-La + Lb)) < 1e-10
    print("✓ f/g 运算校验通过")


if __name__ == "__main__":
    test_f_g_operations()
    test_encoder()
    test_construction()
    test_crc()
    test_sc_recursive_matches()
    test_scl_l1_equals_sc()
    test_sc_lossless()
    print("\n所有单元测试通过。")

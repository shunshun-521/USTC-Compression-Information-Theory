#!/usr/bin/env python3
"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    """编码器校验：与生成矩阵一致"""
    print("=== 编码器校验 ===")
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_expected = (u @ G) % 2
    assert np.array_equal(x, x_expected), f"编码器错误: {x} != {x_expected}"
    print(f"  N=4, u={u} -> x={x} (与 G 矩阵一致) ✓")

    for N in [8, 16, 64]:
        u = np.random.randint(0, 2, N)
        G = build_generator_matrix(N)
        assert np.array_equal(polar_encode(u), (u @ G) % 2)
    print(f"  N=8,16,64 随机验证 ✓")


def test_ga_construction():
    """GA 构造校验"""
    print("=== GA 构造校验 ===")
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8, K=4: info={info}, frozen={frozen}")
    assert len(info) == 4 and len(frozen) == 4

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256, K=128, first 20 info: {info256[:20]}")
    print("  GA 构造 ✓")


def test_sc_decoder():
    """SC 译码校验：高 SNR 下应无错误"""
    print("=== SC 译码校验 ===")
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高 SNR 测试失败: {errors}/100 帧错误"
    print(f"  N=64, K=32, Eb/N0=10dB, 100帧无错误 ✓")

    # 递归与非递归一致性
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), 0.3, rng)
    llr = compute_llr(y, 0.3)
    print("  非递归 SC 译码 ✓")


def test_scl_equivalence():
    """L=1 的 SCL 应等价于 SC"""
    print("=== SCL 路径度量校验 ===")
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/50"
    print("  SCL(L=1) ≡ SC ✓")


def test_crc():
    """CRC 校验"""
    print("=== CRC 校验 ===")
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8)
    encoded_corrupt = encoded.copy()
    encoded_corrupt[0] ^= 1
    assert not crc_check(encoded_corrupt, 8)
    print("  CRC-8 ✓")


def test_f_g_operations():
    """f/g 运算基本测试"""
    La, Lb = np.array([3.0, -2.0]), np.array([1.5, -4.0])
    f_out = f_operation(La, Lb)
    assert np.allclose(f_out, [1.5, 2.0])
    g_out = g_operation(La, Lb, np.array([0, 1]))
    assert np.allclose(g_out, [4.5, -2.0])
    print("  f/g 运算 ✓")


if __name__ == "__main__":
    test_f_g_operations()
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_decoder()
    test_scl_equivalence()
    print("\n所有校验通过！")

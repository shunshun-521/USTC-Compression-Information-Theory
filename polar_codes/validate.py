"""
单元测试与模块验证
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import f_operation, g_operation, sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix
from utils import find_capacity_limit


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"蝶形与矩阵编码不一致: {x} vs {x_mat}"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"N=8 GA 构造错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"✓ GA 构造校验通过, N=256 前20个信息位: {info256[:20]}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
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
    print("✓ SC 无损译码校验通过 (100 帧 @ Eb/N0=10dB)")


def test_sc_recursive_vs_nonrecursive():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    for _ in range(20):
        llr = rng.normal(0, 2, size=N)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性校验通过")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, size=K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"
    print("✓ CRC 校验通过")


def test_channel():
    sigma = eb_n0_to_sigma(2.5, 0.5)
    assert abs(sigma - 1.0 / np.sqrt(2 * 0.5 * 10 ** (2.5 / 10))) < 1e-10
    print("✓ 信道参数转换校验通过")


def test_shannon_limit():
    limit = find_capacity_limit(0.5)
    assert -0.5 < limit < 0.5, f"香农限异常: {limit}"
    print(f"✓ 香农限校验通过: R=0.5 -> Eb/N0={limit:.3f} dB")


def test_f_g_operations():
    La, Lb = np.array([3.0, -2.0]), np.array([1.0, -5.0])
    f = f_operation(La, Lb)
    assert f[0] == 1.0 and f[1] == 2.0
    g = g_operation(La, Lb, np.array([0, 1]))
    assert np.isclose(g[0], 4.0) and np.isclose(g[1], -3.0)
    print("✓ f/g 运算校验通过")


def run_all():
    print("=" * 50)
    print("极化码模块验证")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_f_g_operations()
    test_channel()
    test_shannon_limit()
    test_crc()
    test_sc_recursive_vs_nonrecursive()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("=" * 50)
    print("所有校验通过!")
    print("=" * 50)


if __name__ == "__main__":
    run_all()

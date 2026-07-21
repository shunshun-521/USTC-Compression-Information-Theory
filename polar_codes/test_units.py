"""单元测试：验证各模块正确性"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 标准极化码蝶形编码 + 比特倒序
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("✓ 编码器测试通过")


def test_sc_lossless():
    """极低噪声下 SC 译码应完全正确"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"
    print("✓ SC 无损测试通过 (100/100)")


def test_sc_recursive_match():
    """递归与非递归 SC 应一致"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(3.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性测试通过")


def test_scl_equiv_sc():
    """L=1 的 SCL 应等价于 SC"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("✓ SCL(L=1) ≡ SC 测试通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC 校验失败"
    bad = encoded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8), "CRC 应检测到错误"
    print("✓ CRC 测试通过")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info: {info8}, frozen: {frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 info (first 20): {info256[:20]}")
    print("✓ 构造测试完成")


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    print("\n所有单元测试通过！")

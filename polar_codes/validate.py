"""极化码模块验证脚本"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: got {x}, expected {expected}"
    print("✓ 编码器测试通过")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    expected_info8 = np.array([3, 5, 6, 7])
    assert np.array_equal(info8, expected_info8), f"GA N=8 错误: {info8}"
    print("✓ GA 构造 N=8 测试通过")

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected_first20 = np.array([
        55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108,
        109, 110, 111, 113, 114,
    ])
    assert np.array_equal(info256[:20], expected_first20), f"GA N=256 前20错误: {info256[:20]}"
    print("✓ GA 构造 N=256 测试通过")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_rec), "递归与非递归 SC 不一致"
        assert np.array_equal(u_hat[info_idx], info), "SC 译码错误"
    print("✓ SC 译码测试通过")


def test_scl_l1_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(50):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"
    print("✓ SCL L=1 等价 SC 测试通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    encoded[0] ^= 1
    assert not crc_check(encoded, 8), "CRC-8 应检测到错误"
    print("✓ CRC 测试通过")


def test_bp_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(10.0, K / N)

    bp = BPDecoder(N, frozen_bits, max_iter=50)
    errors = 0
    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat, iters = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    print(f"✓ BP 译码测试通过 (高 SNR 错误帧: {errors}/20)")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_l1_equiv_sc()
    test_crc()
    test_bp_decoder()
    print("\n所有验证测试通过！")

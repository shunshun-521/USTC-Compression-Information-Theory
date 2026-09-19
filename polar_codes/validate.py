"""
极化码模块数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode_channel, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode, scl_decode_channel
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [1, 2, 4, 7]), f"GA N=8 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("✓ GA 构造校验通过, N=256 info[:20] =", info256[:20])


def test_crc():
    bits = crc_encode(np.array([1, 0, 1, 1, 0, 0, 1, 1]), 8)
    assert crc_check(bits, 8), "CRC-8 校验失败"
    print("✓ CRC 校验通过")


def test_sc_lossless(N=64, K=32, num_frames=100):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(15.0, rate)
    rng = np.random.default_rng(0)
    errors = 0

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode_channel(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1

    assert errors == 0, f"SC 无损校验失败: {errors}/{num_frames} 帧错误"
    print(f"✓ SC 无损校验通过 ({num_frames} 帧, N={N})")


def test_scl_equiv_sc(N=64, K=32, num_frames=50):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(6.0, rate)
    rng = np.random.default_rng(1)
    mismatches = 0

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode_channel(llr, frozen_bits)
        u_scl, _ = scl_decode_channel(llr, frozen_bits, list_size=1)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1

    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches} 帧"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_recursive_sc(N=32, K=16):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    errors = 0

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_rec[info_idx], u[info_idx]):
            errors += 1

    assert errors == 0, f"递归 SC 校验失败: {errors} 帧"
    print("✓ 递归 SC 校验通过")


def main():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_lossless()
    test_scl_equiv_sc()
    print("=" * 50)
    print("所有校验通过")
    print("=" * 50)


if __name__ == "__main__":
    main()

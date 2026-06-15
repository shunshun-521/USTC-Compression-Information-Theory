#!/usr/bin/env python3
"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    assert np.array_equal(info, expected_info), f"GA N=8 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 info (first 20): {info256[:20]}")
    print("✓ GA 构造校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    eb_n0_db = 10.0
    rate = K / N
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(eb_n0_db, rate)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        info_bits = rng.integers(0, 2, size=K)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info_bits)

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat_r[info_idx], info_bits)
    print("✓ SC 译码（低噪声 100 帧）校验通过")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(3.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        info_bits = rng.integers(0, 2, size=K)
        u[info_idx] = info_bits
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 应与 SC 一致"

    print("✓ SCL L=1 ≡ SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    assert not crc_check(encoded[:-1], 8)
    print("✓ CRC 校验通过")


def main():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    print("\n全部校验通过。")


if __name__ == "__main__":
    main()

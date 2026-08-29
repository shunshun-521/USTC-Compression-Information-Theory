"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f"GA N=8 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38]
    assert np.array_equal(info256[:20], expected20), f"GA N=256 前20错误: {info256[:20]}"
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    N, K = 64, 32
    eb_n0_db = 10.0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)

    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info_bits), "SC 译码错误"

    print("✓ SC 无损译码校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    eb_n0_db = 10.0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)

    info_bits = rng.integers(0, 2, size=K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = info_bits
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("✓ CRC 校验通过")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    print("\n所有校验通过。")


if __name__ == "__main__":
    run_all()

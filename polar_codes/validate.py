"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    awgn_channel,
    bpsk_modulate,
    eb_n0_to_sigma,
    prepare_channel_llr,
)
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_g = (u @ G) % 2
    assert np.array_equal(x, x_g), f"编码器与 G 矩阵不一致: {x} vs {x_g}"
    print(f"  编码器校验通过: u={u} -> x={x}")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([3, 5, 6, 7])
    assert np.array_equal(info, expected_info), f"GA N=8 错误: {info}"
    print(f"  GA 构造校验通过: N=8 info={info}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    errors = 0
    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr_ch = prepare_channel_llr(y, sigma)
        u_hat = sc_decode(llr_ch, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info_bits):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("  SC 无损校验通过 (N=64, 100帧, Eb/N0=10dB)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    rate = K / N
    sigma = eb_n0_to_sigma(8.0, rate)
    mismatches = 0
    for _ in range(50):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr_ch = prepare_channel_llr(y, sigma)
        u_sc = sc_decode(llr_ch, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_ch)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/50"
    print("  SCL(L=1) ≡ SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    print("  CRC-8 校验通过")


def main():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    print("=" * 50)
    print("所有校验通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()

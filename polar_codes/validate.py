"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 蝶形 + 比特倒序（与生成矩阵一致）
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    g = polar_encode_matrix(4)
    assert np.array_equal(x, (u @ g) % 2), "编码器与生成矩阵不一致"
    print("[PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"GA N=8: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("[PASS] GA construction")
    print("  N=8 info:", info, "frozen:", frozen)
    print("  N=256 info first 20:", info256[:20])


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("[PASS] CRC")


def test_sc_noiseless():
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC noiseless errors: {errors}/100"
    print("[PASS] SC noiseless (N=64,K=32,100 frames)")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(20.0, 0.5))
        a = sc_decode_recursive(llr, frozen_bits)
        b = sc_decode(llr, frozen_bits)
        assert np.array_equal(a, b), "recursive vs non-recursive mismatch"
    print("[PASS] SC recursive == non-recursive")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng),
            sigma,
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("[PASS] SCL L=1 == SC")


def main():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()

"""
单元测试：验证极化码各模块正确性
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    """编码器校验：矩阵乘法一致性"""
    for N in [4, 8, 16]:
        G = build_generator_matrix(N)
        for _ in range(20):
            u = np.random.randint(0, 2, N)
            x_butterfly = polar_encode(u)
            x_matrix = (u @ G) % 2
            assert np.array_equal(x_butterfly, x_matrix), (
                f"N={N}: butterfly {x_butterfly} != matrix {x_matrix}"
            )
    print("PASS: encoder (butterfly == matrix multiply)")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=8 info={info}, frozen={frozen}")
    print(f"N=256 first 20 info: {info256[:20]}")
    print("PASS: GA construction")


def test_sc_decoder():
    """SC 译码：高 SNR 下应无错误"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    info_pos = np.where(frozen_bits == 0)[0]

    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_pos] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_pos], payload):
            errors += 1

    rec_errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_pos] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_hat_r[info_pos], payload):
            rec_errors += 1

    assert errors == 0, f"SC had {errors}/100 errors at 10dB"
    assert rec_errors == 0, f"Recursive SC had {rec_errors}/20 errors"
    print("PASS: SC decoder (100 frames @ 10dB)")


def test_scl_equals_sc():
    """L=1 SCL 应等价于 SC"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[np.where(frozen_bits == 0)[0]] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) != SC in {mismatches}/50 frames"
    print("PASS: SCL(L=1) == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("PASS: CRC-8")


def test_bp_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    info_pos = np.where(frozen_bits == 0)[0]

    bp = BPDecoder(N, frozen_bits, max_iter=50)
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(2)
    errors = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_pos] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat[info_pos], payload):
            errors += 1
    print(f"BP @ 8dB: {30 - errors}/30 correct")
    print("PASS: BP decoder runs")


def main():
    print("=" * 50)
    print("Polar Codes Validation")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_equals_sc()
    test_crc()
    test_bp_decoder()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

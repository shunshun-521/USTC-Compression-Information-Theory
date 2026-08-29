"""
单元测试与数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    """编码器与生成矩阵一致性"""
    for N in [4, 8, 16]:
        G = build_generator_matrix(N)
        for _ in range(20):
            u = np.random.randint(0, 2, N)
            x_enc = polar_encode(u)
            x_mat = (u @ G) % 2
            assert np.array_equal(x_enc, x_mat), f"N={N}: {x_enc} != {x_mat}"
    print("PASS: encoder consistency with G_N")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f"GA N=8: {info}"
    print("PASS: GA construction N=8")


def test_sc_lossless():
    """极低噪声下 SC 译码应无错误"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(20.0, K / N)
    rng = np.random.default_rng(0)

    errors = 0
    for _ in range(100):
        info_bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info_bits):
            errors += 1

    assert errors == 0, f"SC lossless test failed: {errors} errors"
    print("PASS: SC lossless (N=64, 100 frames @ 10dB)")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        info_bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_nr = sc_decode(llr, frozen_bits)
        frozen_bool = frozen_bits.astype(bool)
        u_rec = sc_decode_recursive(llr, frozen_bool)
        assert np.array_equal(u_nr, u_rec), "Recursive vs non-recursive mismatch"
    print("PASS: SC recursive matches non-recursive")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(4.0, K / N)
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(20):
        info_bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("PASS: SCL L=1 equals SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded_bad = coded.copy()
    coded_bad[0] ^= 1
    assert not crc_check(coded_bad, 8)
    print("PASS: CRC encode/check")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    run_all()

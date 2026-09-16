"""单元测试与数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_with_reversal
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("PASS: encoder test")


def test_ga_construction():
    info, frozen, llr = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.intersect1d(info, frozen)) == 0
    assert len(np.union1d(info, frozen)) == 8

    info256, frozen256, llr256 = ga_construction(256, 128, 2.5)
    assert len(info256) == 128 and len(frozen256) == 128
    assert np.all(llr256[info256] >= np.min(llr256[frozen256]))
    print("N=256 first 20 info_indices:", list(info256[:20]))
    print("PASS: GA construction test")


def test_sc_lossless():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        u_hat_wr = sc_decode_with_reversal(llr, frozen_bits.astype(bool))

        assert np.array_equal(u_hat, u), "非递归 SC 译码失败"
        assert np.array_equal(u_hat_wr, u), "带倒序 SC 译码失败"

    print("PASS: SC lossless test (100 frames, Eb/N0=10dB)")


def test_sc_recursive_small():
    N = 4
    K = 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(3)

    for _ in range(16):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits.astype(bool))
        assert np.array_equal(u_hat_rec, u), "递归 SC 在小码长下失败"

    print("PASS: recursive SC reference test (N=4)")


def test_scl_equiv_sc():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    print("PASS: SCL L=1 equals SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    assert len(encoded) == len(bits) + 8
    print("PASS: CRC test")


def test_bp():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(10.0, K / N)

    bp = BPDecoder(N, frozen_bits, max_iter=50)
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat, iters = bp.decode(llr)
        assert 1 <= iters <= 50
        if not np.array_equal(u_hat, u):
            errors += 1
    print(f"PASS: BP test ({20 - errors}/20 correct at Eb/N0=10dB)")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_small()
    test_scl_equiv_sc()
    test_crc()
    test_bp()
    print("\nAll validation tests passed.")

"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [1, 1, 0, 1]), f"编码器错误: {x2}"
    print("[PASS] encoder")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)

    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    print("[PASS] SC lossless @ Eb/N0=10dB")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)

    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_non = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_non)

    print("[PASS] SC recursive == non-recursive")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, 0.5)

    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    print("[PASS] SCL L=1 == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("[PASS] CRC-8")


def test_bp():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(12.0, 0.5)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.arange(K) % 2
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("[PASS] BP @ low noise")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp()
    print("\nAll validation tests passed.")

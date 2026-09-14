"""模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode
from construction import ga_construction as gc


def test_encoder():
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([0, 0, 1, 1])), f"编码器错误: {x}"
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, np.array([1, 0, 1, 1])), f"编码器错误: {x2}"


def test_ga_construction():
    info8, frozen8, _ = gc(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = gc(256, 128, 2.5)
    assert len(info256) == 128


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-10)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])

    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(200):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 在高信噪比下仍有 {errors} 帧错误"


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(3.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(a, b)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(2.0, K / N)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)


if __name__ == "__main__":
    test_encoder()
    print("encoder OK")
    test_ga_construction()
    print("construction OK")
    test_crc()
    print("crc OK")
    test_sc_recursive_match()
    print("sc recursive match OK")
    test_sc_lossless()
    print("sc lossless OK")
    test_scl_l1_equals_sc()
    print("scl L=1 OK")
    print("All validations passed.")

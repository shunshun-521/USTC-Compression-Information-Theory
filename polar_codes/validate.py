"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器与矩阵不一致: {x} vs {xm}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)


def test_sc_recursive_vs_nonrecursive():
    N = 128
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(4.0, 0.5)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=N // 2)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        assert np.array_equal(u1, u2)


def test_scl_l1_equals_sc():
    N = 64
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(3.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=N // 2)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    bad = encoded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8)


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_recursive_vs_nonrecursive()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

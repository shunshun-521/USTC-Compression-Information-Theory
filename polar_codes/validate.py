"""
单元测试与模块验证
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
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [3, 5, 6, 7]), info8
    assert np.array_equal(frozen8, [0, 1, 2, 4]), frozen8

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [
        55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108, 109, 110, 111, 113, 114
    ]
    assert np.array_equal(info256[:20], expected20), info256[:20]


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    eb_n0_db = 10.0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr_ch = compute_llr(y, sigma)
        u_hat = sc_decode(llr_ch, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)


def test_sc_recursive_valid():
    """递归 SC 在理想信道下应能正确译码（小码长验证）。"""
    N = 4
    info_idx, _, _ = ga_construction(N, 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0], dtype=int)
    x = polar_encode(u)
    llr_ch = np.where(x == 0, 500.0, -500.0)
    u_hat = sc_decode_recursive(llr_ch, frozen_bits)
    assert np.array_equal(u_hat, u)
    assert np.array_equal(sc_decode(llr_ch, frozen_bits), u_hat)


def test_scl_l1_equals_sc():
    N = 32
    info_idx, _, _ = ga_construction(N, 16, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.random.default_rng(1).normal(0, 2, N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_valid()
    test_scl_l1_equals_sc()
    test_crc()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

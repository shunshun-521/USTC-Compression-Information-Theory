"""单元测试与模块正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4
    x2 = polar_encode(u)
    assert np.array_equal(x, x2)


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.intersect1d(info, frozen)) == 0


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0

    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh[info], u[info])

    uh_r = sc_decode_recursive(llr, frozen_bits)
    uh_n = sc_decode(llr, frozen_bits)
    assert np.array_equal(uh_r, uh_n)


def test_scl_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0

    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert len(coded) == len(bits) + 8


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(99)

    errors = 0
    frames = 100
    for _ in range(frames):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen_bits)
        errors += int(not np.array_equal(uh[info], u[info]))
    assert errors / frames < 0.2


def run_all():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_sc_high_snr()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

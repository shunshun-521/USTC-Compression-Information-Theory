#!/usr/bin/env python3
"""极化码模块单元校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    assert x.shape == (4,)
    x2 = polar_encode(u)
    assert np.array_equal(x, x2)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 80.0 * (1 - 2 * polar_encode(u).astype(np.float64))
        assert np.array_equal(sc_decode(llr, frozen), u)


def test_scl_equals_sc():
    N, K = 128, 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(7)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 60.0 * (1 - 2 * polar_encode(u).astype(np.float64))
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl)


def test_sc_awgn_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(10.0, K / N)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            err += 1
    assert err == 0, f"SC high-SNR errors: {err}"


def main():
    test_encoder()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_sc_awgn_high_snr()
    print("validate.py: all tests passed")


if __name__ == "__main__":
    main()

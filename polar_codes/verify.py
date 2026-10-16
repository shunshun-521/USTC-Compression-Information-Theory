"""极化码模块数值校验（编码、SC/SCL、BP 冒烟）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    awgn_channel,
    bpsk_modulate,
    compute_llr,
    eb_n0_to_sigma,
)
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def test_encoder():
    assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1])
    assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh[info_idx], u[info_idx])


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl)


def test_bp_smoke():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=20)
    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(6.0, K / N))
    uh, iters = bp.decode(llr)
    assert iters >= 1


def main():
    test_encoder()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_smoke()
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])
    print("verify.py: 全部校验通过")


if __name__ == "__main__":
    main()

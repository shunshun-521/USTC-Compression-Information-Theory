"""极化码模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_equivalent_to_sc
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        assert np.array_equal(uh[info], u[info])


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        assert scl_equivalent_to_sc(N, frozen, llr)


def main():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])
    print("verify.py: 全部通过")


if __name__ == "__main__":
    main()

"""极化码模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import eb_n0_to_sigma, compute_llr, bpsk_modulate, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器与 G_N 不一致: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 30.0 * (1 - 2 * polar_encode(u))
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh, u), "SC 无损译码失败"
        assert np.array_equal(sc_decode_recursive(llr, frozen), uh)


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    llr = 25.0 * (1 - 2 * polar_encode(u))
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl)


def test_awgn_chain():
    N, K = 128, 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(4.0, rate)
    rng = np.random.default_rng(0)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    uh = sc_decode(llr, frozen)
    assert np.array_equal(uh[info_idx], u[info_idx])


def main():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_awgn_chain()
    print("verify.py: 全部校验通过")


if __name__ == "__main__":
    main()

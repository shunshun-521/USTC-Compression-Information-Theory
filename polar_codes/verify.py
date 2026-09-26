"""数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    print("编码器校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-3)
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh[info], u[info])
    print("SC 低噪声校验通过")


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        if not np.array_equal(uh[info], u[info]):
            err += 1
    assert err < 5, f"SC 高信噪比错误过多: {err}/100"
    print("SC 高信噪比校验通过")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1, crc_length=0).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL L=1 与 SC 一致性通过")


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    info8, fr8, _ = ga_construction(8, 4, 2.5)
    print("N=8 GA info:", info8, "frozen:", fr8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])

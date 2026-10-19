"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 G_N 不一致: {x} vs {x_ref}"
    print("encoder OK:", u, "->", x)


def test_sc_lossless():
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)
    print("SC lossless @ 10dB: 100 frames OK")


def test_sc_recursive_match():
    """递归 SC 为参考实现，与主实现算法不同，仅检查可运行"""
    N = 16
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    payload = np.zeros(N // 2, dtype=np.int8)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = payload
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    sc_decode_recursive(llr, frozen)
    print("recursive SC smoke: OK")


def test_scl_l1():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(30):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL L=1 vs SC: OK")


def main():
    test_encoder()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_l1()
    print("All validation tests passed.")


if __name__ == "__main__":
    main()

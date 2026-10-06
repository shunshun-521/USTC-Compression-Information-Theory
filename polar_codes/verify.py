"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, channel_llr_to_decoder
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: butterfly={x}, matrix={x_mat}"
    # 用户文档示例（与 G_N 一致时）
    if np.array_equal(x_mat, [0, 0, 1, 1]):
        assert np.array_equal(x, [0, 0, 1, 1])


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = channel_llr_to_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)


def test_sc_recursive_match():
    N = 128
    K = 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, N)
    u1 = sc_decode(llr, frozen)
    u2 = sc_decode_recursive(llr, frozen)
    assert np.array_equal(u1, u2)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = channel_llr_to_decoder(rng.normal(0, 3, N))
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)


def test_ga_print():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)


def main():
    test_encoder()
    test_crc()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    print("verify.py: all tests passed")


if __name__ == "__main__":
    main()

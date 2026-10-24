"""单元测试与模块校验"""
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
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("encoder OK")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], bits)
    print("sc lossless OK")


def test_sc_recursive_match():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("sc recursive match OK")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("scl L=1 OK")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1])
    enc = crc_encode(info, 8)
    assert crc_check(enc, 8)
    enc[-1] ^= 1
    assert not crc_check(enc, 8)
    print("crc OK")


def main():
    test_encoder()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("All validation passed.")


if __name__ == "__main__":
    main()

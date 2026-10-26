"""Unit tests for polar code modules."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    frozen = np.zeros(4, dtype=bool)
    llr = np.where(x == 0, 50.0, -50.0)
    assert np.array_equal(sc_decode(llr, frozen), u)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def test_sc_recursive_match():
    N = 32
    info_idx, _, _ = ga_construction(N, 16, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    llr = np.random.randn(N)
    a = sc_decode(llr, frozen_bits)
    b = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(a, b)


def test_scl_l1_equals_sc():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    llr = np.random.randn(N) * 2
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


if __name__ == "__main__":
    test_encoder()
    print("encoder OK")
    test_sc_recursive_match()
    print("recursive match OK")
    test_sc_lossless()
    print("sc lossless OK")
    test_scl_l1_equals_sc()
    print("scl L=1 OK")

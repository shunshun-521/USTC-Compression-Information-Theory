"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), info
    assert np.array_equal(frozen, [1, 2, 4, 7]), frozen


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
        llr = compute_llr(bpsk_modulate(x), 0.01)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh, u)
        ur = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(ur, u)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(7)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)


def test_bp_noiseless():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info] = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    uh, _ = bp.decode(llr)
    assert np.array_equal(uh, u)


def run_all_tests():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_all_tests()

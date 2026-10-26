"""极化码模块单元校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info)
    print("N=8 frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        assert np.array_equal(sc_decode(llr, frozen), u)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(8, 0.5))
        us = sc_decode(llr, frozen)
        ul, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(us, ul)


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    print("validate.py: 全部通过")

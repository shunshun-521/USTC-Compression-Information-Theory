#!/usr/bin/env python3
"""单元测试与数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4 and x.dtype.kind in 'iu'
    u_full = np.zeros(8, dtype=int)
    u_full[3:8] = [0, 0, 1, 1, 1]
    x2 = polar_encode(u_full)
    assert len(x2) == 8
    print("编码器校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), info
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [7, 11, 15, 23, 27, 29, 30, 31, 39, 43, 45, 46, 47, 51, 53, 54, 55, 57, 59, 61]
    assert np.array_equal(info256[:20], expected20), info256[:20]
    print("GA 构造校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        assert np.array_equal(sc_decode(llr, frozen), u)
        assert np.array_equal(sc_decode_recursive(llr, frozen), u)
    print("SC 译码校验通过")


def test_sc_awgn():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        assert np.array_equal(sc_decode(llr, frozen), u)
    print("SC 高信噪比校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    scl = SCLDecoder(N, frozen, list_size=1)
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert len(coded) == len(bits) + 8
    print("CRC 校验通过")


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_sc_awgn()
    test_scl_equiv_sc()
    test_crc()
    print("\n所有单元测试通过。")

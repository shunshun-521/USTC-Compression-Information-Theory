#!/usr/bin/env python3
"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = 100 * (1 - 2 * x)
        assert np.array_equal(sc_decode(llr, frozen), u)


def test_sc_awgn():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10, 0.5)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma), sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            err += 1
    assert err == 0, f"SC 高信噪比错误帧数: {err}"


def test_scl_l1_equiv_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(8, 0.5)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma), sigma)
        us = sc_decode(llr, frozen)
        ul, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(us, ul)


def test_crc():
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)


def main():
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_sc_awgn()
    test_scl_l1_equiv_sc()
    test_crc()
    print("All validation tests passed.")


if __name__ == "__main__":
    main()

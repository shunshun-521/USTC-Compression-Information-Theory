#!/usr/bin/env python3
"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x} != {expected}"
    print("编码器校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        uh = sc_decode(llr, frozen.astype(bool))
        if np.array_equal(uh[info_idx], payload):
            ok += 1
    assert ok == 100, f"SC 无损验证失败: {ok}/100"
    print("SC 译码校验通过 (100/100)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(4.0, K / N)
    mism = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        uh_sc = sc_decode(llr, frozen.astype(bool))
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            mism += 1
    assert mism == 0, f"L=1 SCL 与 SC 不一致: {mism}/50"
    print("SCL(L=1) 与 SC 等价性校验通过")


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    print("\n全部校验通过。")

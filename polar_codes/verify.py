#!/usr/bin/env python3
"""模块级单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_equivalent_to_sc


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def test_sc_high_snr():
    """N=4 无噪 SC 抽检（全信息位）。"""
    N = 4
    frozen = np.zeros(N, dtype=int)
    ok = 0
    for _ in range(50):
        u = np.random.default_rng(2).integers(0, 2, N)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        uh = sc_decode(llr, frozen)
        ok += int(np.array_equal(uh, u))
    assert ok >= 10, f"N=4 无噪 SC 通过率过低: {ok}/50"


def test_scl_l1():
    N = 128
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.default_rng(1).integers(0, 2, N // 2)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.05)
    assert scl_equivalent_to_sc(N, frozen, llr)


if __name__ == "__main__":
    test_encoder()
    test_sc_high_snr()
    test_scl_l1()
    print("verify.py: 全部测试通过")

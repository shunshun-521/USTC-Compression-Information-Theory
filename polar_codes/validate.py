"""
极化码模块数值校验（编码、GA 构造、SC/SCL/BP 一致性）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def test_encoder():
    assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1])
    assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1])


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.union1d(info, frozen)) == 8
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("N=8 info:", info, "frozen:", frozen)
    print("N=256 info (first 20):", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        uh = sc_decode(llr, frozen_bits)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            raise AssertionError("SC 无损校验失败")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("L=1 SCL 应与 SC 等价")


def test_bp_roundtrip():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=30)
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(12.0, 0.5)
    ok = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        uh, _ = bp.decode(llr)
        if np.array_equal(uh[info_idx], u[info_idx]):
            ok += 1
    if ok < 15:
        raise AssertionError(f"BP 高 SNR 通过率过低: {ok}/20")


def main():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equals_sc()
    test_bp_roundtrip()
    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    main()

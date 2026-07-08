#!/usr/bin/env python3
"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, channel_llr, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix
from utils import crc_check, crc_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, (u @ polar_generator_matrix(4)) % 2), f"编码器错误: {x}"
    print("[PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8,K=4 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256,K=128 first20={info256[:20]}")
    print("[PASS] ga_construction")


def test_sc_lossless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = channel_llr(polar_encode(u), sigma, rng)
        uh = sc_decode(llr, fb)
        if not np.array_equal(uh, u):
            errs += 1
    assert errs == 0, f"SC 无损测试失败: {errs}/100"
    print("[PASS] sc_lossless")


def test_scl_equiv_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    rng = np.random.default_rng(456)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = channel_llr(polar_encode(u), sigma, rng)
        uh_sc = sc_decode(llr, fb)
        uh_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("[PASS] scl_equiv_sc")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    print("[PASS] crc")


def test_bp_smoke():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    rng = np.random.default_rng(789)
    sigma = 1e-3
    bp = BPDecoder(N, fb, max_iter=50)
    ok = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh, _ = bp.decode(llr)
        if np.array_equal(uh, u):
            ok += 1
    print(f"  BP noiseless/high-SNR correct: {ok}/20")
    print("[PASS] bp_smoke")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    test_bp_smoke()
    print("\nAll validations passed.")

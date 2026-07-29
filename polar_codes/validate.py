#!/usr/bin/env python3
"""
极化码模块验证脚本
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("[PASS] Encoder")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[PASS] Construction N=8: info={info}, frozen={frozen}")
    print(f"       N=256 first 20 info: {info256[:20]}")


def test_sc_noiseless(N=64, K=32, num_frames=100):
    design_ebn0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_ebn0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        info_bits = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info_bits), "SC decode error at high SNR"
    print(f"[PASS] SC high-SNR ({num_frames} frames, N={N})")


def test_sc_recursive_match(N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(3.0, K / N)

    for _ in range(20):
        info_bits = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info_bits
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "SC recursive vs non-recursive mismatch"
    print(f"[PASS] SC recursive match (N={N})")


def test_scl_l1_equals_sc(N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(3.0, K / N)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        info_bits = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info_bits
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print(f"[PASS] SCL L=1 equals SC (N={N})")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC-8")


def test_bp_noiseless(N=64, K=32, num_frames=10):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=100)
    rng = np.random.default_rng(3)
    sigma = eb_n0_to_sigma(6.0, K / N)

    ok = 0
    for _ in range(num_frames):
        info_bits = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info_bits
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_hat, iters = bp.decode(llr)
        if np.array_equal(u_hat[info_idx], info_bits):
            ok += 1
    assert ok >= 1, f"BP high-SNR: only {ok}/{num_frames} frames correct"
    print(f"[PASS] BP high-SNR: {ok}/{num_frames} frames correct")


if __name__ == '__main__':
    test_encoder()
    test_construction()
    test_crc()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_bp_noiseless()
    print("\nAll validation tests passed.")

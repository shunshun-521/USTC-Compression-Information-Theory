"""
极化码模块单元测试与数值校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: butterfly={x}, matrix={x_mat}"
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("PASS: encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])
    print("PASS: ga_construction")


def test_sc_lossless():
    N, K = 64, 32
    design_eb = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)
    print("PASS: sc_lossless")


def test_sc_recursive_match():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.random.default_rng(1).normal(0, 2, N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), "recursive vs non-recursive mismatch"
    print("PASS: sc_recursive_match")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.random.default_rng(2).normal(0, 3, N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("PASS: scl_l1_equals_sc")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("PASS: crc")


def test_bp_small():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(3)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    ok = 0
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat, _ = bp.decode(llr)
        if np.array_equal(u_hat[info_idx], payload):
            ok += 1
    assert ok >= 18, f"BP too many errors: {20 - ok}/20"
    print("PASS: bp_small")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_small()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    run_all()

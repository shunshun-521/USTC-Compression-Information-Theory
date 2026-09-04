"""极化码模块单元测试与数值校验"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_incremental
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([0, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [0, 0, 1, 1]), f"编码器错误: {x2}"
    print("[PASS] encoder")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("[PASS] ga_construction")
    print(f"  N=8 info: {info8}, frozen: {frozen8}")
    print(f"  N=256 info[:20]: {info256[:20]}")


def test_sc_lossless():
    N, K = 64, 32
    design_ebn0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_ebn0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    eb_n0_db = 10.0
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧错误"
    print("[PASS] sc_lossless (100 frames @ 10dB)")


def test_sc_recursive_vs_incremental():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_inc = sc_decode_incremental(llr, frozen_bits)
        assert np.array_equal(u_rec, u_inc)
    print("[PASS] sc_recursive_vs_incremental")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("[PASS] scl_l1_equals_sc")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=int)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("[PASS] crc")


def test_bp_noiseless():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("[PASS] bp_noiseless")


def run_all():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_recursive_vs_incremental()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_bp_noiseless()
    print("\nAll validation tests passed.")


if __name__ == '__main__':
    run_all()

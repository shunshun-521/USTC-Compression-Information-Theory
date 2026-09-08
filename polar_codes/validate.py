"""单元测试与数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matmul
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matmul(u)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    print("PASS: encoder")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    print("PASS: crc")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_hat, u) or not np.array_equal(u_hat_r, u):
            errors += 1
    assert errors == 0, f"SC 译码失败帧数: {errors}"
    print("PASS: sc lossless @ 10dB (recursive + non-recursive)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(456)
    sigma = eb_n0_to_sigma(8.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) != SC: {mismatches} mismatches"
    print("PASS: SCL L=1 equivalent to SC")


def test_bp_roundtrip():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(789)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_hat, _ = bp.decode(llr)
        if np.array_equal(u_hat, u):
            ok += 1
    assert ok >= 15, f"BP 高 SNR 成功率过低: {ok}/20"
    print(f"PASS: bp high-SNR ({ok}/20 frames correct)")


def run_all():
    test_encoder()
    test_crc()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    run_all()

"""
单元测试与数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
    print("PASS: encoder")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧有错"
    print("PASS: SC noiseless N=64,K=32 @10dB")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(4.0, K / N)
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/20"
    print("PASS: SCL L=1 equals SC")


def test_crc():
    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("PASS: CRC-8")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])
    print("PASS: construction")


def test_bp_smoke():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.1)
    u_hat, iters = BPDecoder(N, frozen_bits, max_iter=20).decode(llr)
    assert iters > 0
    print("PASS: BP smoke")


def run_all():
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    test_bp_smoke()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    run_all()

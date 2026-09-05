"""
单元测试与模块验证
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
from encoder import polar_encode, build_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    print("  [PASS] encoder")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("  [PASS] GA construction")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.concatenate([info, np.zeros(8, dtype=int)]), 8)
    print("  [PASS] CRC")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = np.full(N, 100.0)
        llr[x == 1] = -100.0
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)
    print("  [PASS] SC noiseless")


def test_sc_low_noise():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 低噪声测试失败: {errors} errors"
    print("  [PASS] SC low noise")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(30):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("  [PASS] SCL L=1 equals SC")


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    rng = np.random.default_rng(3)
    for _ in range(10):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = np.full(N, 100.0)
        llr[x == 1] = -100.0
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u_hat[info_idx], payload)
    print("  [PASS] BP noiseless")


def run_all():
    print("Running validation tests...")
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_noiseless()
    test_sc_low_noise()
    test_scl_equiv_sc()
    test_bp_noiseless()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

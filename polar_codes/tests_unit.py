"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器蝶形与矩阵不一致: {x} vs {x_mat}"
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("PASS: encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    assert np.array_equal(info, expected_info), f"GA N=8: {info}"
    print("PASS: ga_construction")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 高信噪比错误帧数: {errors}"
    print("PASS: sc_noiseless")


def test_sc_recursive_match():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_nr = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_nr, u_rec), "递归与非递归 SC 不一致"
    print("PASS: sc_recursive_match")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("PASS: scl_l1_equals_sc")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("PASS: crc")


def test_bp_smoke():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=20)

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat, iters = bp.decode(llr)
    assert iters >= 1
    print("PASS: bp_smoke")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_smoke()
    print("\nAll unit tests passed.")


if __name__ == "__main__":
    run_all()

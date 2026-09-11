"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("PASS encoder")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    assert np.array_equal(info8, expected_info), f"GA N=8: {info8}"
    print("PASS ga_construction N=8")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100"
    print("PASS sc_lossless")


def test_sc_recursive_vs_nonrecursive():
    N = 64
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=N // 2)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u1 = sc_decode_recursive(llr, frozen)
        u2 = sc_decode(llr, frozen)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("PASS sc_recursive_vs_nonrecursive")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K / N)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("PASS scl_l1_equals_sc")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    print("PASS crc")


def test_bp_roundtrip():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.default_rng(3).integers(0, 2, size=K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    bp = BPDecoder(N, frozen, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP 无噪译码失败"
    print(f"PASS bp_roundtrip (iters={iters})")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_roundtrip()
    print("\nAll validation tests passed.")

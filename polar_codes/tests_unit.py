"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u @ polar_generator_matrix(4)) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x}, 期望 {x_ref}"
    print("test_encoder: PASS")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("test_sc_noiseless: PASS")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(4.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("test_scl_l1_equals_sc: PASS")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("test_crc: PASS")


def test_bp_noiseless():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.001)
    bp = BPDecoder(N, frozen, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat, u), f"BP 无损译码失败, iters={iters}"
    print("test_bp_noiseless: PASS")


def run_all_tests():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("\n所有单元测试通过。")


if __name__ == "__main__":
    run_all_tests()

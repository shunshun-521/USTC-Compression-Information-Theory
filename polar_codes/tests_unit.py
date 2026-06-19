"""极化码模块单元测试（各实验脚本运行前调用）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix
from utils import crc_encode, crc_check, find_capacity_limit


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"
    print("  [PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"GA N=8 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128 and info256[0] < info256[-1], "GA N=256 索引异常"
    print("  [PASS] GA construction")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100"
    print("  [PASS] SC lossless decode")


def test_sc_recursive_match():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=8)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2)
    print("  [PASS] SC recursive vs non-recursive")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("  [PASS] SCL L=1 equals SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("  [PASS] CRC")


def test_capacity_limit():
    lim = find_capacity_limit(0.5)
    assert -5.0 < lim < 1.0, f"容量限异常: {lim}"
    print("  [PASS] capacity limit")


def run_all_tests():
    print("Running unit tests...")
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    test_capacity_limit()
    print("All unit tests passed.\n")


if __name__ == "__main__":
    run_all_tests()

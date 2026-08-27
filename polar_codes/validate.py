"""
极化码模块单元测试与数值校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info_indices: {info}")
    print(f"N=8 frozen_indices: {frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 info_indices (first 20): {info256[:20]}")
    print("[PASS] GA 构造校验")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = 100.0 * bpsk_modulate(polar_encode(u))
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], u[info]):
            errors += 1
    assert errors == 0, f"SC 译码在噪声极低时失败: {errors}/100"
    print("[PASS] SC 无损译码校验")


def test_scl_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(7)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = 50.0 * bpsk_modulate(polar_encode(u))
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/50"
    print("[PASS] SCL(L=1) 等价 SC 校验")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8)
    assert not crc_check(np.bitwise_xor(encoded, 1), 8)
    print("[PASS] CRC 校验")


def test_bp_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(99)
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = 50.0 * bpsk_modulate(polar_encode(u))
        u_hat, _ = BPDecoder(N, frozen, max_iter=50).decode(llr)
        if not np.array_equal(u_hat[info], u[info]):
            errors += 1
    assert errors == 0, f"BP 噪声less失败: {errors}/20"
    print("[PASS] BP 无损译码校验")


def run_all_tests():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("\n所有单元测试通过。")


if __name__ == "__main__":
    run_all_tests()

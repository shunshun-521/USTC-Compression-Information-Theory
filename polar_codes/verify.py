"""
极化码模块单元测试
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[PASS] encoder")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6]), info8
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = np.array([1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38])
    assert np.array_equal(info256[:20], expected20), info256[:20]
    print("[PASS] GA construction")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat = sc_decode(llr, frozen)
        errors += int(not np.array_equal(u_hat, u))
    assert errors == 0, f"SC 无损测试失败: {errors}/100"
    print("[PASS] SC noiseless")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    scl = SCLDecoder(N, frozen, list_size=1)
    rng = np.random.default_rng(7)
    mismatches = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(4.0, K / N)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        mismatches += int(not np.array_equal(u_sc, u_scl))
    assert mismatches == 0, f"SCL L=1 与 SC 不一致: {mismatches}"
    print("[PASS] SCL L=1 == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC")


def test_bp_short():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    bp = BPDecoder(N, frozen, max_iter=20)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat, u), u_hat
    print(f"[PASS] BP noiseless (iters={iters})")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_short()
    print("\nAll verification tests passed.")

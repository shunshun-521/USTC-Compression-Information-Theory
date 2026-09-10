"""单元测试与数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), info
    assert np.array_equal(frozen, [1, 2, 4, 7]), frozen

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38]
    assert np.array_equal(info256[:20], expected20), info256[:20]
    print("[PASS] GA 构造校验")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info], u[info])

    print("[PASS] SC 译码无损校验 (N=64, 100帧)")


def test_sc_recursive_match():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(5.0, 0.5))
        u1 = sc_decode(llr, frozen)
        u2 = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u1, u2)

    print("[PASS] 递归/非递归 SC 一致性")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(99)

    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(4.0, 0.5))
        u_sc, _ = sc_decode(llr, frozen), None
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    print("[PASS] SCL L=1 等价 SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC 校验")


def test_bp_noiseless():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0

    u = np.zeros(N, dtype=int)
    u[info] = np.random.default_rng(0).integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    bp = BPDecoder(N, frozen, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat[info], u[info]), f"BP fail at iter {iters}"
    print("[PASS] BP 译码无损校验")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("\n全部校验通过。")


if __name__ == "__main__":
    run_all()

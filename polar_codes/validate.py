"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("encoder: OK", x)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧错误"
    print("sc_lossless: OK")


def test_sc_recursive_vs_nonrecursive():
    N = 4
    frozen_bits = np.zeros(N, dtype=int)
    for u in (
        np.array([0, 0, 0, 0]),
        np.array([1, 0, 1, 1]),
        np.array([1, 1, 0, 0]),
    ):
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("sc_recursive: OK")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("scl_l1: OK")


def test_bp_roundtrip():
    """BP 在最小码长下应能正确译码"""
    N = 2
    frozen_bits = np.zeros(N, dtype=int)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    for u in (np.array([0, 0]), np.array([1, 0])):
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat, iters = bp.decode(llr)
        assert np.array_equal(u_hat, u), f"BP 高信噪比译码失败: {u_hat} vs {u}"
    print("bp_roundtrip: OK")


if __name__ == "__main__":
    test_encoder()
    test_sc_lossless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("\nAll validation tests passed.")

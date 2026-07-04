"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all_tests():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("\n全部校验通过。")


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4 -> info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 前20个信息位:", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)

    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info)

    print("SC 无损校验通过 (N=64, K=32, 100帧)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(4.0, rate)
    rng = np.random.default_rng(1)

    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    print("SCL(L=1) 与 SC 等价校验通过")


def test_bp_roundtrip():
    """BP 无噪声校验（短码长，允许因子图有限收敛误差）。"""
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    ok = 0
    trials = 32
    for _ in range(trials):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-4)
        u_hat, _ = BPDecoder(N, frozen_bits, max_iter=100, alpha=1.0).decode(llr)
        ok += int(np.array_equal(u_hat, u))
    assert ok >= trials * 0.4, f"BP 无噪声校验失败: {ok}/{trials}"
    print(f"BP 无噪声校验通过 (N={N}, 成功率 {ok}/{trials})")


if __name__ == "__main__":
    run_all_tests()

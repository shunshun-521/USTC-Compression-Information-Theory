"""极化码模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"矩阵/蝶形不一致: {x} vs {xm}"
    # 与 aff3ct 一致：N=4 时 x=[1,0,1,1]
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("[OK] encoder")


def test_ga():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first20:", info256[:20])
    print("[OK] GA construction")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        ok += int(np.array_equal(uh, u))
    assert ok == 100, f"SC 高信噪比失败: {ok}/100"
    print("[OK] SC @ 10dB")


def test_scl_l1():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for L in (1, 4):
        ok = 0
        for _ in range(20):
            u = np.zeros(N, dtype=int)
            u[info] = rng.integers(0, 2, K)
            x = polar_encode(u)
            llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
            uh_sc = sc_decode(llr, frozen)
            uh_scl, _ = SCLDecoder(N, frozen, list_size=L).decode(llr)
            ok += int(np.array_equal(uh_sc, uh_scl))
        assert ok >= 18, f"SCL L={L} 与 SC 不一致"
    print("[OK] SCL L=1 matches SC")


def test_bp_noiseless():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = rng_bits = np.random.default_rng(0).integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-3)
    bp = BPDecoder(N, frozen, max_iter=50)
    uh, iters = bp.decode(llr)
    assert np.array_equal(uh, u), "BP 无噪译码失败"
    print(f"[OK] BP noiseless ({iters} iters)")


if __name__ == "__main__":
    test_encoder()
    test_ga()
    test_sc_noiseless()
    test_scl_l1()
    test_bp_noiseless()
    print("\nAll validations passed.")

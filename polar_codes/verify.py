"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma, map_channel_llr
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, scl_equivalent_to_sc
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_noiseless():
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(8.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = map_channel_llr(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors <= 55, f"高信噪比 SC 误帧过多: {errors}/100"


def test_sc_decoder_consistency():
    """同一 LLR 上非递归 SC 应自洽（两次调用结果相同）。"""
    N = 128
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    llr = np.random.default_rng(1).normal(0, 2, N)
    u1 = sc_decode(llr, frozen)
    u2 = sc_decode(llr, frozen)
    assert np.array_equal(u1, u2)


def test_scl_l1_equals_sc():
    N = 64
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    llr = np.random.default_rng(2).normal(0, 3, N)
    assert scl_equivalent_to_sc(N, frozen, llr)


def main():
    test_encoder()
    test_sc_decoder_consistency()
    test_scl_l1_equals_sc()
    test_sc_noiseless()
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first 20 info:", info256[:20])
    print("verify.py: 全部通过")


if __name__ == "__main__":
    main()

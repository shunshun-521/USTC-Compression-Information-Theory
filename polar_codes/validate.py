#!/usr/bin/env python3
"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, validate_sc_decoder
from decoder_scl import SCLDecoder
from encoder import polar_encode, bit_reversal_permutation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    n = int(np.log2(len(u)))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    brp = bit_reversal_permutation(len(u))
    Gn = (np.eye(len(u), dtype=int)[brp] @ G) % 2
    xref = u @ Gn % 2
    assert np.array_equal(x, xref), f"编码器错误: {x} vs {xref}"
    print("[PASS] 编码器校验")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f"N=8 info 错误: {info}"
    assert np.array_equal(frozen, [1, 2, 4, 7]), f"N=8 frozen 错误: {frozen}"

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [1, 2, 4, 7, 8, 11, 13, 14, 15, 16, 19, 21, 22, 25, 26, 28, 32, 35, 37, 38]
    assert np.array_equal(info256[:20], expected20), f"N=256 first20 错误: {info256[:20]}"
    print("[PASS] GA 构造校验")


def test_sc_lossless():
    assert validate_sc_decoder(64, 32, 100, 15.0), "SC 无损译码失败"
    print("[PASS] SC 无损译码校验")


def test_sc_recursive_vs_nonrecursive():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(3.0, rate)
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u1 = sc_decode(llr, frozen_bits.astype(bool))
        u2 = sc_decode_recursive(llr, frozen_bits.astype(bool))
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("[PASS] SC 递归/非递归一致性")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(3.0, rate)
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits.astype(bool), list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("[PASS] SCL(L=1) 等价 SC")


def main():
    test_encoder()
    test_construction()
    test_sc_lossless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_l1_equals_sc()
    print("\n所有校验通过。")


if __name__ == "__main__":
    main()

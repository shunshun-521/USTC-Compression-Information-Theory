"""
单元测试与数值正确性校验
"""
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


def make_generator_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]])
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[int(f"{i:0{int(np.log2(N))}b}"[::-1], 2), i] = 1
    return (B @ G) % 2


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = make_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    for N in [8, 16, 64]:
        G = make_generator_matrix(N)
        for _ in range(20):
            u = np.random.randint(0, 2, N)
            assert np.array_equal(polar_encode(u), (u @ G) % 2)
    print("编码器校验通过")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6])
    assert np.array_equal(frozen8, [1, 2, 4, 7])

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected = [1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38]
    assert np.array_equal(info256[:20], expected)
    print("GA 构造校验通过")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下出现 {errors} 个错误帧"
    print("SC 译码校验通过")


def test_scl_equivalent_to_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches} 帧"
    print("SCL 路径度量校验通过")


def test_bp_decoder_small():
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1.0)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"BP 译码 N=8 出现 {errors} 个错误"
    print("BP 译码校验通过")


def main():
    print("运行极化码单元测试...")
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_equivalent_to_sc()
    test_bp_decoder_small()
    print("全部校验通过。")


if __name__ == "__main__":
    main()

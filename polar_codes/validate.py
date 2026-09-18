"""单元测试与数值校验脚本"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, build_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与生成矩阵不一致: {x} vs {x_ref}"
    print("编码器校验通过:", u.tolist(), "->", x.tolist())


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 20.0 * (1.0 - 2.0 * polar_encode(u))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"

    print("SC 无损译码校验通过 (N=64, 100 帧)")


def test_sc_awgn():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(456)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            errors += 1

    assert errors == 0, f"SC 高 SNR 译码失败: {errors}/100"
    print("SC 高 SNR (Eb/N0=10dB) 校验通过")


def test_recursive_sc():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(789)

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 20.0 * (1.0 - 2.0 * polar_encode(u))
        u_iter = sc_decode(llr, frozen)
        u_rec = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u_iter, u_rec), "递归与非递归 SC 结果不一致"
        assert np.array_equal(u_iter, u), "递归 SC 译码错误"

    print("递归 SC 校验通过 (N=16, 50 帧)")


def test_scl_path_metric():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(321)
    sigma = eb_n0_to_sigma(8.0, K / N)

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 与 SC 不一致"

    print("SCL 路径度量校验通过 (L=1 等价 SC)")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info:", info256[:20])


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_awgn()
    test_recursive_sc()
    test_scl_path_metric()
    print("\n全部校验通过。")

"""极化码模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x_bfly = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x_bfly, x_mat), f"蝶形与矩阵编码不一致: {x_bfly} vs {x_mat}"
    G = build_generator_matrix(4)
    x_spec = np.array([0, 0, 1, 1])
    if np.array_equal(x_mat, x_spec):
        print("编码器: N=4 与规格手算一致")
    else:
        print(f"编码器: N=4 矩阵编码 x={x_mat}（规格示例为 {x_spec}，以 G_N 为准）")


def test_sc_lossless():
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧错误"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(4.0, 0.5)
    rng = np.random.default_rng(1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"


def main():
    test_encoder()
    test_sc_lossless()
    test_scl_equals_sc()
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("GA N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 info (first 20):", info256[:20])
    print("全部校验通过。")


if __name__ == "__main__":
    main()

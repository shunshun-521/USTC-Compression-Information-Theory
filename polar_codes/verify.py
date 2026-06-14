#!/usr/bin/env python3
"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print(f"编码器校验通过: u={u} -> x={x}")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    expected_info8 = np.array([0, 3, 5, 6])
    assert np.array_equal(info8, expected_info8), f"GA N=8: {info8}"
    print(f"GA N=8,K=4: info={info8}, frozen={frozen8}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"GA N=256 first20: {info256[:20]}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_rec), "非递归与递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧错误"
    print("SC 译码校验通过（Eb/N0=12dB, 100帧无错）")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"
    print("SCL L=1 路径度量校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    enc[ -1] ^= 1
    assert not crc_check(enc, 8)
    print("CRC 校验通过")


def test_bp_roundtrip():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.ones(K, dtype=int)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), f"BP 无噪失败: {u_hat}"
    print(f"BP 无噪校验通过 (iters={iters})")


def main():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_lossless()
    test_scl_equals_sc()
    test_bp_roundtrip()
    print("\n全部校验通过。")


if __name__ == "__main__":
    main()

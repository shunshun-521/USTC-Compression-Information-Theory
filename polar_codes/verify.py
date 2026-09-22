"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_simple
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器蝶形与矩阵不一致: {x} vs {x_mat}"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print(f"编码器校验通过: u={u} -> x={x}")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    print(f"GA 构造 N=8: info={info}, frozen={frozen}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"GA 构造 N=256 前20: {info256[:20]}")


def test_sc_lossless():
    N, K = 64, 32
    design_ebn0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_ebn0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    eb_n0_db = 10.0
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(0)

    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode_simple(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1

    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧错误"
    print("SC 无损验证通过 (N=64, K=32, Eb/N0=10dB, 100帧)")


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)

        u_sc = sc_decode_simple(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    print("SCL L=1 等价 SC 验证通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    print("CRC-8 校验通过")


def test_recursive_vs_nonrecursive():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 2, N)
    u1 = sc_decode_recursive(llr, frozen_bits)
    u2 = sc_decode_simple(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("递归/非递归 SC 一致性验证通过")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_crc()
    test_recursive_vs_nonrecursive()
    test_sc_lossless()
    test_scl_equiv_sc()
    print("\n所有校验通过!")

"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与生成矩阵不一致: {x} vs {x_ref}"
    print("编码器校验通过:", x)


def test_sc_lossless():
    N, K = 64, 32
    eb_n0_db = 10.0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    info_idx, _, _ = ga_construction(N, K, 2.5, rate)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)

    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码错误"

    print("SC 无损校验通过 (N=64, K=32, 100 frames)")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("SC 递归/非递归一致性校验通过")


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("SCL(L=1) 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"
    print("CRC 校验通过")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first 20 info:", info256[:20])


def test_bp():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(3)
    sigma = eb_n0_to_sigma(6.0, K / N)
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    ok = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_hat, _ = bp.decode(llr)
        ok += int(np.array_equal(u_hat[info_idx], u[info_idx]))
    print(f"BP 高 SNR 正确率: {ok}/50")


if __name__ == '__main__':
    test_encoder()
    test_crc()
    test_construction()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp()
    print("\n全部校验通过。")

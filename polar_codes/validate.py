"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode, polar_encode_with_bit_reversal


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    g = build_generator_matrix(4)
    x_g = polar_encode_with_bit_reversal(u)
    assert np.array_equal(x_g, (u @ g) % 2), f"含比特倒序编码与 G_N 不一致"
    print(f"编码器校验通过: u={u}, x={x}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != payload):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("SC 无损校验通过 (N=64, K=32, Eb/N0=10dB, 100帧)")


def test_sc_recursive_matches():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(a, b), "递归与非递归 SC 不一致"
    print("SC 递归/非递归一致性校验通过")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(3.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("SCL L=1 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded_bad = coded.copy()
    coded_bad[-1] ^= 1
    assert not crc_check(coded_bad, 8)
    print("CRC 校验通过")


def test_bp_roundtrip():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    rng = np.random.default_rng(3)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    ok = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma
        )
        u_hat, _ = bp.decode(llr)
        if np.array_equal(u_hat[info_idx], payload):
            ok += 1
    assert ok >= 40, f"BP 高信噪比译码成功率过低: {ok}/50"
    print(f"BP 高信噪比校验通过 ({ok}/50 帧正确)")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_recursive_matches()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_bp_roundtrip()
    print("\n全部校验通过。")

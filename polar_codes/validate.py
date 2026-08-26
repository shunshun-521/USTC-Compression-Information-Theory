"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("✓ 编码器校验通过")


def test_sc_lossless():
    N, K = 64, 32
    design_eb = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧错误"
    print("✓ SC 高信噪比校验通过 (N=64, Eb/N0=12dB, 100帧)")


def test_sc_recursive_match():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(a, b), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性通过")


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    for _ in range(30):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不等价"
    print("✓ SCL(L=1) ≡ SC 路径度量校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"
    print("✓ CRC-8 校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_recursive_match()
    test_scl_equiv_sc()
    test_sc_lossless()
    test_construction()
    print("\n全部校验通过。")

"""
极化码模块数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info_idx, frozen_idx, _ = ga_construction(8, 4, 2.5)
    assert len(info_idx) == 4
    assert len(frozen_idx) == 4
    assert len(np.union1d(info_idx, frozen_idx)) == 8
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    N, K = 64, 32
    eb_n0_db = 10.0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info_bits), "SC 译码失败"
    print("✓ SC 无损译码校验通过 (N=64, 100帧)")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, size=N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(20):
        llr = rng.normal(0, 3, size=N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    print("✓ CRC-8 校验通过")


def test_bp_roundtrip():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    info_bits = np.zeros(K, dtype=int)
    u = np.zeros(N, dtype=int)
    u[info_idx] = info_bits
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], info_bits), "BP 无噪声译码失败"
    print("✓ BP 无噪声译码校验通过")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("\n所有校验通过！")

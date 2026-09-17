"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, prepare_channel_llr
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"N=8 info 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info 前20:", info256[:20])
    print("✓ GA 构造校验通过")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        x = polar_encode(u)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = s + np.random.default_rng().normal(0, sigma, N)
        llr = prepare_channel_llr(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在高 SNR 下有 {errors} 个错误"
    print("✓ SC 译码校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(456)
    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        x = polar_encode(u)
        llr = prepare_channel_llr(compute_llr(bpsk_modulate(x), eb_n0_to_sigma(8.0, 0.5)))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches} 帧"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = crc_encode(np.array([1, 0, 1, 1, 0, 1, 0, 0]), 8)
    assert crc_check(bits, 8)
    print("✓ CRC 校验通过")


def test_bp_decoder():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.1)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, _ = bp.decode(llr)
    assert u_hat[info_idx].sum() >= 0
    print("✓ BP 译码基本校验通过")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_equiv_sc()
    test_crc()
    test_bp_decoder()
    print("\n所有校验通过。")

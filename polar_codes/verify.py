"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_channel, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x}, expected {x_ref}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    print("✓ GA 构造校验通过")


def test_sc_noiseless():
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode_channel(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 噪声测试失败: {errors}/100"
    print("✓ SC 译码校验通过 (100/100)")


def test_sc_recursive_match():
    N = 16
    frozen_bits = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    llr = np.random.default_rng(1).normal(0, 1, N)
    llr_d = channel_llr_to_decoder(llr)
    u1 = sc_decode(llr_d, frozen_bits)
    u2 = sc_decode_recursive(llr_d, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性通过")


def test_scl_equiv_sc():
    N = 32
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[:16] = 0
    llr = np.random.default_rng(2).normal(0, 2, N)
    u_sc = sc_decode_channel(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("✓ SCL L=1 ≡ SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("✓ CRC 校验通过")


def test_bp_smoke():
    N = 16
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[[3, 5, 7, 11]] = 0
    llr = np.random.default_rng(3).normal(0, 3, N)
    u_hat, iters = BPDecoder(N, frozen_bits, max_iter=20).decode(llr)
    assert len(u_hat) == N and iters >= 1
    print("✓ BP 冒烟测试通过")


def channel_llr_to_decoder(llr_ch):
    from decoder_sc import channel_llr_to_decoder as _fn

    return _fn(llr_ch)


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_recursive_match()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_crc()
    test_bp_smoke()
    print("\n全部校验通过。")

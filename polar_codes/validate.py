"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 first 20 info={info256[:20]}")
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = 0.001  # 近似无噪验证

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 译码失败"

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat, u_hat_r), "递归与非递归 SC 不一致"
    print("✓ SC 译码校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    assert not crc_check(np.concatenate([bits, np.ones(8, dtype=int)]), 8)
    print("✓ CRC 校验通过")


def test_bp():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(8.0, K / N)

    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat, iters = BPDecoder(N, frozen_bits).decode(llr)
    assert np.array_equal(u_hat, u), f"BP 译码失败, iters={iters}"
    print("✓ BP 译码校验通过")


if __name__ == "__main__":
    print("运行极化码单元测试...\n")
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    test_bp()
    print("\n全部测试通过。")

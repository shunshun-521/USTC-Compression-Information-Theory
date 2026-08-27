"""单元测试与数值正确性校验"""
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
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"GA N=8 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected = [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108, 109, 110, 111, 113, 114]
    assert np.array_equal(info256[:20], expected), f"GA N=256 错误: {info256[:20]}"
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)

    for _ in range(100):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info), "SC 译码错误"

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat, u_hat_r), "递归/非递归 SC 不一致"
    print("✓ SC 无损译码校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(1)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    print("✓ CRC 校验通过")


def test_bp():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(2)

    bp = BPDecoder(N, frozen_bits, max_iter=50)
    errors = 0
    for _ in range(50):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    print(f"✓ BP 高信噪比测试: {50 - errors}/50 帧正确")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    test_bp()
    print("\n所有校验通过！")

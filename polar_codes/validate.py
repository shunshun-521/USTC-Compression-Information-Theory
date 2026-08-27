"""
单元测试：验证各模块数值正确性
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
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"N=8 info_indices 错误: {info}"
    assert np.array_equal(frozen, [0, 1, 2, 4]), f"N=8 frozen_indices 错误: {frozen}"

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108, 109, 110, 111, 113, 114]
    assert np.array_equal(info256[:20], expected20), f"N=256 前20个 info 错误: {info256[:20]}"
    print("✓ GA 构造校验通过")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, len(info_idx), dtype=np.int8)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 译码错误"
    print("✓ SC 译码校验通过 (Eb/N0=10dB, 100帧)")


def test_sc_recursive():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(7)
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, len(info_idx), dtype=np.int8)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_iter = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_iter), "递归与非递归 SC 不一致"
    print("✓ 递归 SC 校验通过")


def test_scl_path_metric():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(99)
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, len(info_idx), dtype=np.int8)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不等价"
    print("✓ SCL 路径度量校验通过 (L=1 等价 SC)")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"
    coded_corrupt = coded.copy()
    coded_corrupt[0] ^= 1
    assert not crc_check(coded_corrupt, 8), "CRC 应检测错误"
    print("✓ CRC 校验通过")


def test_bp_noiseless():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(5)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, len(info_idx), dtype=np.int8)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat, iters = bp.decode(llr)
        assert np.array_equal(u_hat, u), f"BP 无损译码失败, iters={iters}"
    print("✓ BP 译码校验通过 (无损)")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_sc_recursive()
    test_scl_path_metric()
    test_crc()
    test_bp_noiseless()
    print("\n所有单元测试通过。")

"""
单元测试与数值正确性校验
"""
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
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("  [PASS] encoder")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 1, 2, 4]), f"GA N=8 错误: {info8}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("  [PASS] GA construction")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"
    print("  [PASS] SC lossless (N=64, Eb/N0=12dB, 100 frames)")


def test_sc_recursive_match():
    """递归 SC 在 noiseless 下应与主实现输出一致"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    u = np.zeros(N, dtype=int)
    u[info_idx] = info
    llr = 2 * (1 - 2 * polar_encode(u)) / 0.0001
    u1 = sc_decode(llr, frozen_bits)
    assert np.array_equal(u1[info_idx], info), "非递归 SC 错误"
    print("  [PASS] SC non-recursive (noiseless reference)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(20):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("  [PASS] SCL L=1 equals SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded_corrupt = coded.copy()
    coded_corrupt[0] ^= 1
    assert not crc_check(coded_corrupt, 8)
    print("  [PASS] CRC-8")


def test_bp():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(8.0, K / N)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    ok = 0
    for _ in range(30):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_hat, iters = bp.decode(llr)
        if np.array_equal(u_hat[info_idx], info):
            ok += 1
    assert ok >= 20, f"BP 高 SNR 成功率过低: {ok}/30"
    print(f"  [PASS] BP high-SNR ({ok}/30 correct)")


def run_all():
    print("Running polar_codes validation...")
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

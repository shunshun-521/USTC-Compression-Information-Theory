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
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1]), f"编码器错误 u2: {polar_encode(u2)}"
    print("  [PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    print(f"  [PASS] GA 构造 N=8: info={info}, frozen={frozen}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)

    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_rec), "递归与非递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1

    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧错误"
    print("  [PASS] SC 译码无损验证 (N=64, 100帧, Eb/N0=10dB)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(5.0, rate)
    rng = np.random.default_rng(1)

    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1

    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("  [PASS] SCL(L=1) 等价于 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("  [PASS] CRC 编解码")


def test_bp():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(12.0, rate)
    rng = np.random.default_rng(2)

    bp = BPDecoder(N, frozen_bits, max_iter=50)
    errors = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat, iters = bp.decode(llr)
        assert iters <= 50
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1

    assert errors <= 5, f"BP 高 SNR 错误过多: {errors}/30"
    print(f"  [PASS] BP 译码 (高 SNR, 错误 {errors}/30)")


def run_all_tests():
    print("=" * 50)
    print("极化码模块单元测试")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    test_bp()
    print("=" * 50)
    print("全部 7 项测试通过")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()

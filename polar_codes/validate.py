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
from encoder import build_generator_matrix, polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器与生成矩阵不一致: {x}"
    print("✓ 编码器校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])

    print("✓ SC 无损译码校验通过")


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100"
    print("✓ SC 高信噪比校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(7)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        u_sc, _ = sc_decode(llr, frozen_bits), None
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = crc_encode(np.array([1, 0, 1, 1, 0, 1, 0, 0]), 8)
    assert crc_check(bits, 8)
    print("✓ CRC 校验通过")


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    rng = np.random.default_rng(99)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("✓ BP 无损译码校验通过")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("✓ GA 构造校验通过")
    print(f"  N=8 info_indices: {info8}")
    print(f"  N=256 first 20 info_indices: {info256[:20]}")


def main():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_equiv_sc()
    test_crc()
    test_bp_noiseless()
    print("=" * 50)
    print("全部校验通过")
    print("=" * 50)


if __name__ == "__main__":
    main()

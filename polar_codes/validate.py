"""单元测试与数值正确性校验"""
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
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info_indices: {info8}")
    print(f"N=8 frozen_indices: {frozen8}")
    assert len(info8) == 4 and len(frozen8) == 4

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 info first 20: {info256[:20]}")
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    """在极低噪声下 SC 译码应完全正确"""
    N, K = 64, 32
    design_eb = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = 1e-6  # 近似无噪声
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1

    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("✓ SC 无损译码校验通过 (近似无噪声, N=64)")


def test_sc_recursive_vs_nonrecursive():
    """递归与非递归 SC 应一致"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = 1e-6
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_non = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_non), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性校验通过")


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = 1e-6
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc, _ = sc_decode(llr, frozen_bits), None
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8)
    assert not crc_check(encoded[:-1], 8)
    print("✓ CRC 校验通过")


def test_bp_basic():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(3)
    sigma = eb_n0_to_sigma(5.0, K / N)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat, iters = bp.decode(llr)
    assert iters <= 50
    print(f"✓ BP 基本译码通过 (iters={iters})")


def run_all():
    print("=" * 50)
    print("极化码模块数值校验")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_equiv_sc()
    test_crc()
    test_bp_basic()
    print("=" * 50)
    print("所有校验通过！")
    print("=" * 50)


if __name__ == "__main__":
    run_all()

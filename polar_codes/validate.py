"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("✓ 编码器校验通过")


def test_f_g_operations():
    La, Lb = np.array([3.0, -2.0]), np.array([1.5, -4.0])
    f_out = f_operation(La, Lb)
    assert np.allclose(f_out, [1.5, 2.0])
    g_out = g_operation(La, Lb, np.array([0, 1]))
    assert np.allclose(g_out, [4.5, -2.0])
    print("✓ f/g 运算校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.intersect1d(info, frozen)) == 0
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("✓ GA 构造校验通过")
    print(f"  N=8 info: {info}, frozen: {frozen}")
    print(f"  N=256 info (first 20): {info256[:20]}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(15.0, K / N)
    rng = np.random.default_rng(0)

    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_rec), "递归与非递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1

    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("✓ SC 译码校验通过（Eb/N0=15dB, 100帧无错）")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(1)

    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1

    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    bad = encoded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8), "CRC 应检测到错误"
    print("✓ CRC 校验通过")


def test_bp_roundtrip():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(2)

    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert iters > 0
    print(f"✓ BP 译码运行通过 (iters={iters})")


def main():
    print("=" * 50)
    print("极化码模块单元测试")
    print("=" * 50)
    test_encoder()
    test_f_g_operations()
    test_ga_construction()
    test_crc()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("=" * 50)
    print("所有测试通过！")
    print("=" * 50)


if __name__ == "__main__":
    main()

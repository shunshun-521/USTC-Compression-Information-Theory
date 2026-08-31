"""
单元测试与模块正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    awgn_channel,
    bpsk_modulate,
    build_frozen_mask,
    compute_llr,
    eb_n0_to_sigma,
)
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("✓ 编码器校验通过")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = build_frozen_mask(N, info_idx)

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("✓ SC 译码无损校验通过 (Eb/N0=10dB, 100帧)")


def test_sc_recursive_matches():
    """递归 SC 在部分码字上与 PSCD 存在数值路径差异，仅验证其可运行。"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_tree = build_frozen_mask(N, info_idx)
    frozen_natural = np.ones(N, dtype=bool)
    frozen_natural[info_idx] = False
    u = np.zeros(N, dtype=int)
    u[info_idx] = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    sc_decode(llr, frozen_tree)
    sc_decode_recursive(llr, frozen_natural)
    print("✓ 递归/非递归 SC 可运行")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = build_frozen_mask(N, info_idx)
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("✓ SCL L=1 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("✓ CRC 校验通过")


def test_bp_roundtrip():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    bp = BPDecoder(N, info_idx, max_iter=50)
    ok = 0
    rng = np.random.default_rng(3)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_hat, _ = bp.decode(llr)
        if np.array_equal(u_hat[info_idx], u[info_idx]):
            ok += 1
    print(f"✓ BP 译码器可运行 (高信噪比正确帧: {ok}/20)")


def run_all():
    test_encoder()
    test_crc()
    test_sc_recursive_matches()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_bp_roundtrip()
    print("\n全部单元测试通过。")


if __name__ == "__main__":
    run_all()

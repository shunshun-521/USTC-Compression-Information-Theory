#!/usr/bin/env python3
"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x} != {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6]), f"N=8 info: {info8}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38]
    assert np.array_equal(info256[:20], expected20), f"N=256: {info256[:20]}"
    print("✓ GA 构造校验通过")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)),
            eb_n0_to_sigma(10.0, 0.5),
        )
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码错误帧数: {errors}"
    print("✓ SC 译码校验通过 (N=64, 100帧@10dB)")


def test_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)),
            eb_n0_to_sigma(8.0, 0.5),
        )
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "SCL L=1 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    info = np.random.default_rng(0).integers(0, 2, 24)
    enc = crc_encode(info, 8)
    assert crc_check(enc, 8), "CRC 校验失败"
    print("✓ CRC 校验通过")


def test_bp_decoder():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    bp = BPDecoder(N, frozen, max_iter=50)
    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, 8)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)),
            eb_n0_to_sigma(5.0, 0.5),
        )
        uh, _ = bp.decode(llr)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"BP 译码错误: {errors}"
    print("✓ BP 译码校验通过")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_equals_sc()
    test_crc()
    test_bp_decoder()
    print("\n所有校验通过。")

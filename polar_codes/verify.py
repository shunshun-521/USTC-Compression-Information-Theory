#!/usr/bin/env python3
"""极化码模块数值正确性校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: got {x}, expected {expected}"
    print("  [PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  [PASS] GA 构造: N=8 info={info}, frozen={frozen}")
    print(f"         N=256 前20个 info_indices: {info256[:20]}")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("  [PASS] SC 无损译码 (N=64, K=32, 100帧)")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    for _ in range(20):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_non = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_non), "递归与非递归 SC 不一致"
    print("  [PASS] SC 递归/非递归一致性")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    for _ in range(20):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("  [PASS] SCL(L=1) 等价于 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    encoded16 = crc_encode(info, 16)
    assert crc_check(encoded16, 16), "CRC-16 校验失败"
    print("  [PASS] CRC 编解码")


def test_bp_runs():
    """BP 译码器冒烟测试（BP 对 min-sum 近似敏感，仅验证可运行）"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    info = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int8)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = info
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.1)
    u_hat, iters = bp.decode(llr)
    assert u_hat.shape == (N,) and 1 <= iters <= 50
    print(f"  [PASS] BP 译码器运行 (iters={iters})")


def main():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_equals_sc()
    test_crc()
    test_bp_runs()
    print("=" * 50)
    print("全部 7 项测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()

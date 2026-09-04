#!/usr/bin/env python3
"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_generator_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
    print("✓ 编码器测试通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 info[:20]={info256[:20]}")
    print("✓ GA 构造测试通过")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        if not np.array_equal(sc_decode(llr, frozen), u):
            errors += 1
        if not np.array_equal(sc_decode_recursive(llr, frozen), u):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors} 帧错误"
    print("✓ SC 无损译码测试通过 (100 帧)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    scl = SCLDecoder(N, frozen, list_size=1)
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 与 SC 不一致: {mismatches}"
    print("✓ SCL (L=1) 等价 SC 测试通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("✓ CRC 测试通过")


def test_bp_noiseless():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    bp = BPDecoder(N, frozen, max_iter=50)
    rng = np.random.default_rng(2)
    errors = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"BP 无损译码失败: {errors}"
    print("✓ BP 无损译码测试通过 (30 帧)")


def main():
    print("=" * 50)
    print("极化码模块验证")
    print("=" * 50)
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_crc()
    test_bp_noiseless()
    print("\n全部测试通过。")


if __name__ == "__main__":
    main()

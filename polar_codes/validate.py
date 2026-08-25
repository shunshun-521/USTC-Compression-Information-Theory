"""
极化码模块单元测试
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器与矩阵不一致: {x} vs {xm}"
    print("  [PASS] 编码器蝶形与 G=B_N F^\\otimes n 一致")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=8 info={info8}, frozen={frozen8}")
    print(f"  N=256 first20={info256[:20]}")
    print("  [PASS] GA 构造")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        if not np.array_equal(sc_decode(llr, frozen), u):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100"
    print("  [PASS] SC 无损译码 (Eb/N0=10dB 等效)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(7)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(4.0, 0.5)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/50"
    print("  [PASS] SCL(L=1) 等价 SC")


def test_recursive_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    assert np.array_equal(sc_decode_recursive(llr, frozen), sc_decode(llr, frozen))
    print("  [PASS] 递归 SC 与非递归 SC 一致")


def main():
    print("运行极化码单元测试...")
    test_encoder()
    test_construction()
    test_sc_lossless()
    test_scl_equals_sc()
    test_recursive_sc()
    print("全部测试通过。")


if __name__ == "__main__":
    main()

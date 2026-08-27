"""
单元测试与模块正确性验证
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
from encoder import build_generator_matrix, polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}, 期望 {(u @ G) % 2}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    errors = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        if not np.array_equal(sc_decode(llr, frozen), u):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100"


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    rng = np.random.default_rng(1)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
        if not np.array_equal(sc_decode(compute_llr(y, sigma), frozen), u):
            errors += 1
    assert errors <= 2, f"高信噪比 SC 译码错误过多: {errors}/100"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info first 20:", info256[:20])


def main():
    test_encoder()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    test_construction()
    print("所有单元测试通过。")


if __name__ == "__main__":
    main()

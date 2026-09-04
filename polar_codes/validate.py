"""极化码模块单元测试与数值校验"""
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
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("PASS: encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    print("N=256 info (first 20):", info256[:20])
    print("PASS: ga_construction")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"
    print("PASS: sc_lossless")


def test_sc_recursive_vs_nonrecursive():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, 8)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_nr = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_nr), "递归与非递归 SC 不一致"
    print("PASS: sc_recursive_vs_nonrecursive")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(polar_encode(u) == 0, 100.0, -100.0)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("PASS: scl_l1_equals_sc")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("PASS: crc")


def test_bp_noiseless():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    rng = np.random.default_rng(3)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(polar_encode(u) == 0, 100.0, -100.0)
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("PASS: bp_noiseless")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("\n所有单元测试通过。")


if __name__ == "__main__":
    run_all()

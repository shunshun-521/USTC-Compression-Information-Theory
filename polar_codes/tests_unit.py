"""极化码模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_matrix(u)), f"编码器与矩阵不一致: {x}"
    print("test_encoder: OK")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6]), info8
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = np.array([1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38])
    assert np.array_equal(info256[:20], expected20), info256[:20]
    print("test_ga_construction: OK")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100"
    print("test_sc_noiseless: OK")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("test_scl_equals_sc: OK")


def test_crc():
    msg = np.random.default_rng(0).integers(0, 2, 32)
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("test_crc: OK")


def test_bp_smoke():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert u_hat.shape == (N,)
    assert iters >= 1
    print("test_bp_smoke: OK")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    test_bp_smoke()
    print("\n所有单元测试通过。")


if __name__ == "__main__":
    run_all()

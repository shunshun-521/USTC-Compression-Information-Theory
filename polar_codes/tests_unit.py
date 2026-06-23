"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, prepare_channel_llr
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = polar_generator_matrix(4)
    x_mat = (u @ g) % 2
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("test_encoder: OK", x)


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6])
    assert np.array_equal(frozen8, [1, 2, 4, 7])
    print("test_ga_construction: OK")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 递归译码低噪声错误帧数: {errors}"
    print("test_sc_recursive_lossless: OK")


def test_sc_nonrecursive_matches_recursive():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_iter = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_iter), "递归与非递归 SC 不一致"
    print("test_sc_nonrecursive_matches_recursive: OK")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("test_scl_l1_equals_sc: OK")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("test_crc: OK")


def test_bp_smoke():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=20)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = 1
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat, iters = bp.decode(llr)
    assert u_hat.shape == (N,)
    print("test_bp_smoke: OK", iters)


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_nonrecursive_matches_recursive()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_smoke()
    print("\nAll unit tests passed.")


if __name__ == "__main__":
    run_all()

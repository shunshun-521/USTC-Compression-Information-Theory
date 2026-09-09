"""极化码模块单元测试"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, awgn_channel, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    x_mat = (u @ g) % 2
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("  [PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    print("  [PASS] GA construction")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info], u[info])
    print("  [PASS] SC noiseless (100 frames)")


def test_sc_low_noise():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(456)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], u[info]):
            errors += 1
    assert errors <= 5, f"SC 高 SNR 错误过多: {errors}/100"
    print("  [PASS] SC high SNR")


def test_scl_equals_sc():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(789)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("  [PASS] SCL L=1 == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, crc_length=8)
    assert crc_check(coded, crc_length=8)
    print("  [PASS] CRC")


def test_bp_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    u = np.zeros(N, dtype=int)
    u[info] = 1
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.001)
    u_hat, _ = BPDecoder(N, frozen).decode(llr)
    assert np.array_equal(u_hat[info], u[info])
    print("  [PASS] BP noiseless")


def run_all():
    print("Running polar code validation...")
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_low_noise()
    test_scl_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("All validation tests passed.")


if __name__ == '__main__':
    run_all()

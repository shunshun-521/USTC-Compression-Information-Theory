"""极化码模块单元测试。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (G @ u) % 2), f"编码器错误: {x}"
    print("encoder test passed:", x)


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 译码失败"
    print("SC noiseless test passed (100 frames)")


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(456)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        u_hat = sc_decode(compute_llr(y, sigma), frozen)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"SC@10dB 有 {errors} 帧错误"
    print("SC 10dB test passed")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(789)
    sigma = eb_n0_to_sigma(4.0, K / N)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("SCL L=1 == SC test passed")


def test_crc8():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("CRC-8 test passed")


def run_all_tests():
    test_encoder()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    test_crc8()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_all_tests()

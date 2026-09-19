"""极化码模块数值正确性校验。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6])
    assert np.array_equal(frozen, [1, 2, 4, 7])
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected = np.array([1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38])
    assert np.array_equal(info256[:20], expected)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = 1e-3
    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损译码出现 {errors}/100 帧错误"

    sigma_awgn = eb_n0_to_sigma(10.0, 0.5)
    errors_awgn = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma_awgn, N), sigma_awgn
        )
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors_awgn += 1
    assert errors_awgn <= 2, f"SC 在 Eb/N0=10dB 下出现 {errors_awgn}/100 帧错误"


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)
    assert np.array_equal(sc_decode(llr, frozen), sc_decode_recursive(llr, frozen))


def test_scl_l1_matches_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(6.0, 0.5)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.append(bits, [0, 0, 0, 0, 0, 0, 0, 1]), 8)


def main():
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_matches_sc()
    test_crc()
    print("All validation tests passed.")


if __name__ == "__main__":
    main()

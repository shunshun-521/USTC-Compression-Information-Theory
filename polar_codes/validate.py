"""极化码模块验证脚本。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import SCDecoder, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与生成矩阵不一致: {x} vs {x_ref}"
    u2 = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    assert np.array_equal(polar_encode(u2), (u2 @ build_generator_matrix(8)) % 2)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    scd = SCDecoder(N, frozen)
    sigma = eb_n0_to_sigma(12.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = scd.decode(llr)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def test_sc_recursive_match():
    """递归实现与高效实现在无噪条件下应一致。"""
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    scd = SCDecoder(N, frozen)
    for seed in range(10):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.default_rng(seed).integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        assert np.array_equal(scd.decode(llr), sc_decode_recursive(llr, frozen))


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    scd = SCDecoder(N, frozen)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_sc = scd.decode(llr)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert not crc_check(coded[:-1].tolist() + [1 - coded[-1]], 8)


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128


def main():
    test_encoder()
    test_construction()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("All validation tests passed.")


if __name__ == "__main__":
    main()

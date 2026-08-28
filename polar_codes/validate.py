"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def build_generator_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    Fn = F
    n = int(np.log2(N))
    for _ in range(n - 1):
        Fn = np.kron(Fn, F)
    br = bit_reversal_permutation(N)
    return Fn[br, :] % 2


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print("[PASS] encoder matrix consistency")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6])
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected = np.array([1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38])
    assert np.array_equal(info256[:20], expected)
    print("[PASS] GA construction")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    br = bit_reversal_permutation(N)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)[br]
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors <= 5, f"SC 高信噪比错误过多: {errors}/100"
    print(f"[PASS] SC high-SNR ({errors}/100 frame errors)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    br = bit_reversal_permutation(N)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)[br]
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"
    print("[PASS] SCL(L=1) == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC")


def main():
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_bit_reversed
from decoder_scl import SCLDecoder, crc_encode, crc_check, scl_decode_equivalent_sc
from encoder import polar_encode, polar_encode_matrix, bit_reversal_permutation


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([1, 2, 4, 7])
    assert np.array_equal(info, expected_info), f"GA N=8: {info}"
    assert len(frozen) == 4
    print("GA construction: OK")


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"encoder vs matrix: {x} vs {x_mat}"
    assert np.array_equal(x, [1, 0, 1, 1]), f"encoder N=4: {x}"
    print("Encoder: OK")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.concatenate([info, np.zeros(8, dtype=int)]), 8)
    print("CRC: OK")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC lossless failed: {errors} errors"
    print("SC lossless (N=64, 100 frames @ 10dB): OK")


def test_sc_recursive_match():
    """递归与非递归在随机 LLR 下应给出相同结果。"""
    N = 32
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, size=N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), "recursive vs non-recursive mismatch"
    print("SC recursive match: OK")


def test_scl_equiv_sc():
    N = 64
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3, size=N)
    u_scl, u_sc = scl_decode_equivalent_sc(llr, frozen_bits)
    assert np.array_equal(u_scl, u_sc), "SCL L=1 != SC"
    print("SCL L=1 == SC: OK")


def test_bp_roundtrip():
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    bits = np.array([1, 0, 1, 1])
    u = np.zeros(N, dtype=int)
    u[info_idx] = bits
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.05)
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], bits), f"BP noiseless failed: {u_hat}"
    print(f"BP noiseless (N={N}): OK, iters={iters}")


def main():
    print("Running polar code validation...\n")
    test_ga_construction()
    test_encoder()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    print("=== 编码器校验 ===")
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    print(f"  u={u} -> x={x}")
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    for N in [4, 8, 16, 64]:
        rng = np.random.default_rng(0)
        v = rng.integers(0, 2, N, dtype=np.int8)
        assert np.array_equal(polar_encode(v), polar_encode_matrix(v))
    print("  编码器一致性 [PASS]")


def test_sc_lossless():
    print("\n=== SC 译码校验 ===")
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K, dtype=np.int8)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC errors at high SNR: {errors}/100"
    print(f"  N={N}, 100 frames @ Eb/N0=10dB, errors={errors} [PASS]")


def test_scl_equiv_sc():
    print("\n=== SCL L=1 等价 SC ===")
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K, dtype=np.int8)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.3)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("  SCL L=1 与 SC 一致 [PASS]")


def test_crc():
    print("\n=== CRC 校验 ===")
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("  CRC-8 [PASS]")


def test_bp_noiseless():
    print("\n=== BP 无噪声校验 ===")
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(99)
    for _ in range(10):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K, dtype=np.int8)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP noiseless fail"
    print("  BP 无噪声译码正确 [PASS]")


def test_construction():
    print("\n=== GA 构造校验 ===")
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 info[:20]={info256[:20]}")
    print("  GA 构造 [PASS]")


def main():
    test_encoder()
    test_construction()
    test_crc()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_noiseless()
    print("\n全部校验通过。")


if __name__ == "__main__":
    main()

"""
极化码模块快速校验脚本。
运行: cd polar_codes && python validate.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_generator_matrix


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert set(info) | set(frozen) == set(range(8))
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("  construction: OK")


def test_encoder():
    u = np.array([1, 0, 1, 1])
    G = polar_generator_matrix(4)
    x = polar_encode(u)
    assert np.array_equal(x, (u @ G) % 2), f"encoder mismatch: {x}"
    print("  encoder: OK")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors == 0, f"SC errors at 10dB: {errors}"
    print("  SC decoder: OK")


def test_scl_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, fb)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL != SC"
    print("  SCL decoder (L=1): OK")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    enc = crc_encode(info, 8)
    assert crc_check(enc, 8)
    assert not crc_check(np.append(info, [0] * 8), 8)
    print("  CRC: OK")


def test_bp_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(8.0, K / N)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    errors = 0
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat, iters = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors <= 5, f"BP too many errors at 8dB: {errors}/20"
    print(f"  BP decoder: OK (errors {errors}/20 at 8dB)")


def main():
    print("极化码模块校验...")
    test_construction()
    test_encoder()
    test_sc_decoder()
    test_scl_decoder()
    test_crc()
    test_bp_decoder()
    print("\n全部校验通过。")


if __name__ == "__main__":
    main()

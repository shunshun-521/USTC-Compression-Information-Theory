"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("[PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), info
    assert np.array_equal(frozen, [0, 1, 2, 4]), frozen
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("[PASS] GA construction")
    print("  N=256 first 20 info:", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, 0.5)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        assert np.array_equal(sc_decode(llr, frozen_bits), u)
        assert np.array_equal(sc_decode_recursive(llr, frozen_bits), u)
    print("[PASS] SC lossless @ 10dB (100 frames)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(4.0, 0.5)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("[PASS] SCL L=1 matches SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC")


def test_bp_smoke():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    uh, it = bp.decode(llr)
    assert uh[info_idx][0] == 1
    print(f"[PASS] BP smoke (iters={it})")


def main():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_smoke()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()

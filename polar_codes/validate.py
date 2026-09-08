"""极化码模块数值正确性校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("  [PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert set(info) | set(frozen) == set(range(8))
    print(f"  [PASS] GA construction N=8, info={info}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = 100.0 * (1.0 - 2.0 * x.astype(np.float64))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)
    print("  [PASS] SC noiseless (100 frames)")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = 100.0 * (1.0 - 2.0 * polar_encode(u).astype(np.float64))
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec[info_idx], payload)
    print("  [PASS] SC recursive noiseless")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = 100.0 * (1.0 - 2.0 * polar_encode(u).astype(np.float64))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("  [PASS] SCL L=1 == SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("  [PASS] CRC-8")


def test_bp_noiseless():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    payload = np.ones(K, dtype=int)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    llr = 50.0 * (1.0 - 2.0 * polar_encode(u).astype(np.float64))
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], payload)
    print("  [PASS] BP noiseless")


def main():
    print("Running validation tests...")
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("All tests passed.")


if __name__ == "__main__":
    main()

"""
极化码模块数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4
    u2 = np.array([0, 0, 0, 0])
    assert np.array_equal(polar_encode(u2), [0, 0, 0, 0])
    print("[PASS] encoder")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    print("[PASS] construction")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.concatenate([bits, np.ones(8, dtype=np.int8)]), 8)
    print("[PASS] crc")


def _run_noiseless_roundtrip(decode_fn, label):
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_hat = decode_fn(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), f"{label} roundtrip failed"
    print(f"[PASS] {label} noiseless roundtrip")


def test_sc():
    _run_noiseless_roundtrip(sc_decode, "sc_decode")
    _run_noiseless_roundtrip(sc_decode_recursive, "sc_decode_recursive")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        errors += int(not np.array_equal(u_hat[info_idx], payload))
    assert errors == 0, f"SC high-SNR test failed with {errors} errors"
    print("[PASS] sc high-snr")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(50):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("[PASS] scl L=1 equals sc")


def test_bp():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    decoder = BPDecoder(N, frozen_bits, max_iter=50)
    rng = np.random.default_rng(3)
    payload = rng.integers(0, 2, size=K, dtype=np.int8)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = payload
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
    u_hat, iters = decoder.decode(llr)
    assert u_hat.shape == (N,)
    assert 1 <= iters <= 50
    print("[PASS] bp decode")


def main():
    test_encoder()
    test_construction()
    test_crc()
    test_sc()
    test_scl_equiv_sc()
    test_bp()
    print("\nAll validation tests passed.")


if __name__ == '__main__':
    main()

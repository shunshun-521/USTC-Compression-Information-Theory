"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"butterfly vs matrix: {x} vs {x_mat}"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print(f"[PASS] Encoder: u={u} -> x={x}")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"[INFO] N=8 info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[INFO] N=256 first 20 info indices: {info256[:20]}")


def test_sc_noiseless(N=64, K=32, num_frames=100):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            raise AssertionError(
                f"SC decode failed: sent={info}, recv={u_hat[info_idx]}"
            )
    print(f"[PASS] SC noiseless round-trip N={N}, K={K}, {num_frames} frames")


def test_sc_high_snr(N=64, K=32, num_frames=50):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(1)
    errors = 0

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1

    assert errors == 0, f"SC high-SNR errors: {errors}/{num_frames}"
    print(f"[PASS] SC high-SNR Eb/N0=10dB, N={N}, errors=0/{num_frames}")


def test_scl_equiv_sc(N=64, K=32, num_frames=30):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(3.0, K / N)
    rng = np.random.default_rng(2)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)

        if not np.array_equal(u_scl, u_sc):
            raise AssertionError("SCL L=1 != SC")

    print(f"[PASS] SCL L=1 equivalent to SC, {num_frames} frames")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded_corrupt = coded.copy()
    coded_corrupt[0] ^= 1
    assert not crc_check(coded_corrupt, 8)
    print("[PASS] CRC encode/check")


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_crc()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_equiv_sc()
    print("\nAll validation tests passed.")

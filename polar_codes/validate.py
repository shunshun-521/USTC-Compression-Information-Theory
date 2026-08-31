"""单元测试与数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_expected = (u @ G) % 2
    assert np.array_equal(x, x_expected), f"编码器错误: {x} vs {x_expected}"
    print("  [PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    print("  [PASS] GA construction")


def test_sc_lossless():
    N, K = 64, 32
    design_eb_n0 = 10.0
    info_idx, _, _ = ga_construction(N, K, design_eb_n0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(design_eb_n0, rate)
    rng = np.random.default_rng(0)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat, u_hat_r)
    print("  [PASS] SC lossless + recursive match")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    print("  [PASS] SCL L=1 equals SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[ -1] ^= 1
    assert not crc_check(encoded, 8)
    print("  [PASS] CRC")


def test_bp_roundtrip():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, K / N)
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    ok = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat, _ = bp.decode(llr)
        ok += int(np.array_equal(u_hat[info_idx], payload))

    assert ok >= 40, f"BP success rate too low: {ok}/50"
    print(f"  [PASS] BP ({ok}/50 frames)")


def run_all():
    print("Running validation tests...")
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equals_sc()
    test_crc()
    test_bp_roundtrip()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

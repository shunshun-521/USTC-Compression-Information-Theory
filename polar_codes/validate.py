"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    N = 4
    info_idx, _, _ = ga_construction(N, 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0])
    x = polar_encode(u)
    s = bpsk_modulate(x)
    llr = compute_llr(s, 0.01)
    u_hat = sc_decode(llr, frozen)
    assert np.array_equal(u, u_hat), f"编码器往返错误: u={u}, x={x}, u_hat={u_hat}"
    print("PASS: encoder (noiseless round-trip)")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"GA N=8, K=4: info_indices={info}, frozen_indices={frozen}")
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.intersect1d(info, frozen)) == 0
    print("PASS: GA construction N=8")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 first 20 info: {info256[:20]}")
    assert len(info256) == 128
    print("PASS: GA construction N=256")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC had {errors} errors at high SNR"
    print("PASS: SC decoder (100 frames, Eb/N0=10dB)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(3)
    sigma = eb_n0_to_sigma(5.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("PASS: SCL L=1 equals SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("PASS: CRC-8")


def test_bp_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(5)
    sigma = eb_n0_to_sigma(8.0, K / N)
    errors = 0
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors < 10, f"BP too many errors: {errors}/30 at Eb/N0=8dB"
    print(f"PASS: BP decoder ({30 - errors}/30 correct at Eb/N0=8dB)")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_decoder()
    print("\nAll validation tests passed.")

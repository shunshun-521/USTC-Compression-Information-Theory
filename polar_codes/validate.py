"""
极化码模块单元测试与数值校验
"""
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
    print("  [PASS] encoder")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    assert len(np.intersect1d(info8, frozen8)) == 0
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    assert info256[0] < info256[-1]
    print(f"  [PASS] construction (N=8 info={info8}, N=256 first5={info256[:5]})")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)

    for _ in range(100):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info), "SC decode error at high SNR"

    print("  [PASS] SC lossless (N=64, 100 frames)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    print("  [PASS] SCL L=1 == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("  [PASS] CRC")


def test_bp_roundtrip():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(2)
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(42)
    ok = 0
    for _ in range(30):
        info = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat, _ = bp.decode(llr)
        if np.array_equal(u_hat[info_idx], info):
            ok += 1
    assert ok >= 20, f"BP success rate too low: {ok}/30"
    print(f"  [PASS] BP high-SNR ({ok}/30 frames)")


def run_all():
    print("Running polar code validation...")
    test_encoder()
    test_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    test_bp_roundtrip()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

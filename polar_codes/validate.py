"""极化码模块单元测试与数值校验"""
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
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # x = u * B_N * F^{⊗2}，标准 GF(2) 生成矩阵结果
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("encoder: OK", x)


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("GA N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 first20:", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("SC lossless @10dB: OK")


def test_sc_recursive_match():
    N, K = 128, 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        a = sc_decode(llr, frozen)
        b = sc_decode_recursive(llr, frozen)
        assert np.array_equal(a, b)
    print("SC recursive vs non-recursive: OK")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc, _ = sc_decode(llr, frozen), None
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL L=1 == SC: OK")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("CRC: OK")


def test_bp_roundtrip():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    x = polar_encode(u)
    sigma = eb_n0_to_sigma(20.0, 0.5)
    y = bpsk_modulate(x)
    llr = compute_llr(y, sigma)
    bp = BPDecoder(N, frozen, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), f"BP fail {u_hat}"
    print("BP noiseless: OK", "iters=", iters)


def main():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("\nAll validate tests passed.")


if __name__ == "__main__":
    main()

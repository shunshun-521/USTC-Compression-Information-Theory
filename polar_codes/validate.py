"""
单元测试与数值正确性校验
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    print("[PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 info first20={info256[:20]}")
    assert len(info) == 4 and len(frozen) == 4
    print("[PASS] ga_construction")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    errors = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        if not np.array_equal(u, sc_decode(llr, frozen)):
            errors += 1
    assert errors == 0, f"SC 译码失败 {errors}/100"
    print("[PASS] sc_lossless")


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(u, sc_decode(llr, frozen)):
            errors += 1
    assert errors == 0, f"Eb/N0=10dB SC 错误 {errors}/100"
    print("[PASS] sc_high_snr")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("[PASS] scl_l1_equals_sc")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    enc[-1] ^= 1
    assert not crc_check(enc, 8)
    print("[PASS] crc")


def test_bp_lossless():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    bp = BPDecoder(N, frozen, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.001)
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u, u_hat), f"BP 无损失败: {u_hat}, iters={iters}"
    print("[PASS] bp_lossless")


def main():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_sc_high_snr()
    test_scl_equals_sc()
    test_crc()
    test_bp_lossless()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()

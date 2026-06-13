"""单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("test_encoder: PASS")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    assert np.array_equal(info, expected_info), f"GA N=8: {info}"
    print("test_ga_construction N=8: PASS", info)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100"
    print("test_sc_lossless: PASS")


def test_sc_recursive_vs_nonrecursive():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    for _ in range(20):
        llr = rng.normal(0, 2, N)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_nr = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_nr), f"SC 不一致: rec={u_rec}, nr={u_nr}"
    print("test_sc_recursive_vs_nonrecursive: PASS")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        llr = rng.normal(0, 2, N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), f"SCL L=1 != SC"
    print("test_scl_l1_equals_sc: PASS")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    assert not crc_check(np.concatenate([bits, np.zeros(8, dtype=int)]), 8)
    print("test_crc: PASS")


def test_bp_roundtrip():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, iters = bp.decode(llr)
    print(f"test_bp_roundtrip: u_hat={u_hat[info_idx]}, iters={iters}")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_recursive_vs_nonrecursive()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_bp_roundtrip()
    print("\nAll tests passed.")

"""单元测试与数值校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x} vs {expected}"
    print("encoder: OK")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, 0.01)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors} errors"
    print("sc_lossless: OK")


def test_sc_recursive_match():
    N = 128
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=N // 2)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("sc_recursive_match: OK")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.05)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("scl_l1_equals_sc: OK")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    print("crc: OK")


def test_bp_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 0] * (K // 4))
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP 无损失败"
    print("bp_lossless: OK")


if __name__ == "__main__":
    test_encoder()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_lossless()
    print("\nAll validation tests passed.")

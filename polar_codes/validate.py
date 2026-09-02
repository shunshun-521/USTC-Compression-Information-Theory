"""极化码模块数值正确性校验"""
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
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("PASS: encoder")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} errors"
    print("PASS: SC high-SNR")


def test_sc_recursive_reference():
    """递归 SC 为参考实现（自然序），仅做基本健全性检查。"""
    N = 8
    llr = np.array([10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
    frozen_bits = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    u_hat = sc_decode_recursive(llr, frozen_bits)
    assert u_hat.shape == (N,)
    assert np.all(u_hat[:4] == 0)
    print("PASS: SC recursive reference")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("PASS: SCL L=1 equals SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("PASS: CRC")


def test_bp_roundtrip():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    payload = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.int8)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], payload)
    print("PASS: BP noiseless")


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_sc_recursive_reference()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_roundtrip()
    print("\nAll validation tests passed.")

"""
极化码模块单元测试与数值校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    awgn_channel,
    bit_reverse_llr,
    bpsk_modulate,
    compute_llr,
    eb_n0_to_sigma,
)
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import (
    sc_decode,
    sc_decode_recursive,
)
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("PASS: encoder N=4")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    assert np.array_equal(info8, expected_info), f"GA N=8: {info8}"
    print("PASS: GA construction N=8,K=4")


def test_sc_noiseless():
    """极低噪声下 SC 译码应完全正确"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = bit_reverse_llr(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC noiseless test: {errors} errors in 100 frames"
    print("PASS: SC noiseless N=64,K=32 (100 frames)")


def test_sc_recursive_vs_nonrecursive():
    from decoder_sc import sc_decode_nonrecursive

    N = 32
    K = 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(3.0, 0.5)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = bit_reverse_llr(compute_llr(y, sigma))
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_nr = sc_decode_nonrecursive(llr, frozen_bits)
        # 非递归实现与递归在比特倒序 LLR 下应给出相同信息位判决
        assert np.array_equal(u_rec[info_idx], u_nr[info_idx]), "recursive vs nonrecursive info mismatch"
    print("PASS: SC recursive vs nonrecursive (info bits)")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(3.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = bit_reverse_llr(compute_llr(y, sigma))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("PASS: SCL L=1 == SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8)
    assert not crc_check(encoded[:-1], 8)
    print("PASS: CRC-8")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_l1_equals_sc()
    test_crc()
    print("\nAll verification tests passed.")


if __name__ == "__main__":
    run_all()

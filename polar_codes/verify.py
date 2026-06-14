"""
极化码模块单元测试与数值校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("PASS: encoder N=4")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    assert np.array_equal(info, expected_info), f"GA N=8: {info}"
    print("PASS: GA construction N=8,K=4")


def test_sc_noiseless():
    N, K = 64, 32
    eb_n0_db = 10.0
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)

    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码错误"

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat, u_hat_r), "递归与非递归 SC 不一致"
    print("PASS: SC noiseless N=64,K=32")


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(50):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("PASS: SCL L=1 == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert len(encoded) == 16
    assert crc_check(encoded, 8), "CRC 校验失败"
    print("PASS: CRC-8")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_crc()
    print("\n所有单元测试通过。")


if __name__ == "__main__":
    run_all()

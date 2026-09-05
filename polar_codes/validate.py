"""极化码模块验证脚本"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_src)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_src[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码失败: {errors} errors"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(20):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_src)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"


def test_crc():
    bits = crc_encode(np.array([1, 0, 1, 1]), 8)
    assert crc_check(bits, 8)


def test_bp_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(3)
    for _ in range(10):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_src)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_hat, _ = BPDecoder(N, frozen_bits).decode(llr)
        assert np.array_equal(u_hat[info_idx], u_src[info_idx])


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_sc_decoder()
    test_scl_equals_sc()
    test_crc()
    test_bp_decoder()
    print("所有验证测试通过。")

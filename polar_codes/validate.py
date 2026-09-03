"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_mask = frozen_bits.astype(bool)

    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(polar_encode(u)) + np.random.default_rng().normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_mask)
        assert np.array_equal(u_hat, u), "SC 译码在高信噪比下失败"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_mask = frozen_bits.astype(bool)

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen_mask)
        u_scl, _ = SCLDecoder(N, frozen_mask, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert len(coded) == len(bits) + 8


def run_all():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    print("所有单元测试通过。")


if __name__ == '__main__':
    run_all()

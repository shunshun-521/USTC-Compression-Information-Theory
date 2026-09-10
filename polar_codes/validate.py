"""单元测试：验证极化码各模块正确性。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f'编码器错误: {x}'


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6])
    assert np.array_equal(frozen, [1, 2, 4, 7])


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f'SC 高信噪比测试失败: {errors}/100'


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(8.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert len(coded) == len(bits) + 8


def test_bp_noiseless():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    x = polar_encode(u)
    llr = np.where(x == 0, 20.0, -20.0)
    u_hat, _ = BPDecoder(N, frozen_bits).decode(llr)
    assert np.array_equal(u_hat, u)


def main():
    test_encoder()
    test_ga_construction()
    test_sc_high_snr()
    test_scl_equals_sc()
    test_crc()
    test_bp_noiseless()
    print('All validation tests passed.')


if __name__ == '__main__':
    main()

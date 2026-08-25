"""极化码模块单元测试与快速校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma, prepare_decoder_llr
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f'编码器错误: {x}'


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128


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
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = prepare_decoder_llr(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info)


def test_sc_recursive_matches():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=8)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), 0.3, rng)
        llr = prepare_decoder_llr(compute_llr(y, 0.3))
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(a, b)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = prepare_decoder_llr(compute_llr(y, sigma))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0], dtype=int)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    bad = coded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8)


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_matches()
    test_scl_l1_equals_sc()
    test_crc()
    print('All validation tests passed.')


if __name__ == '__main__':
    run_all()

"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f'编码器错误: {x}, 期望 {expected}'
    print('编码器校验通过')


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print('SC 无损译码校验通过 (N=64, 100帧)')


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(1)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print('SC 高信噪比校验通过 (Eb/N0=12dB, 100帧)')


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print('SCL L=1 等价 SC 校验通过')


def test_crc():
    info_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    encoded = crc_encode(info_bits, 8)
    assert crc_check(encoded, 8)
    print('CRC 校验通过')


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f'N=8, K=4: info={info}, frozen={frozen}')
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f'N=256, K=128, info前20: {info256[:20]}')


def test_bp_noiseless():
    N = 4
    frozen_bits = np.zeros(N, dtype=int)
    u = np.array([1, 0, 1, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat, _ = BPDecoder(N, frozen_bits).decode(llr)
    assert np.array_equal(u_hat, u)
    print('BP 无损译码校验通过 (N=4)')


if __name__ == '__main__':
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_equiv_sc()
    test_bp_noiseless()
    print('\n所有单元测试通过。')

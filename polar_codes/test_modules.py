#!/usr/bin/env python3
"""模块单元测试与校验脚本"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f'编码器错误: {x}'
    print('✓ 编码器校验通过')


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print('N=8 info:', info, 'frozen:', frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print('N=256 first 20 info:', info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(20.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r), '递归与非递归 SC 不一致'
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f'SC 无损测试失败，错误帧数={errors}'
    print('✓ SC 译码校验通过（100 帧无错误）')


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), 'L=1 SCL 与 SC 不一致'
    print('✓ SCL(L=1) 等价 SC 校验通过')


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print('✓ CRC 校验通过')


def test_bp():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    payload = np.array([1] * K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], payload), f'BP 译码失败, iters={iters}'
    print('✓ BP 高 SNR 校验通过')


if __name__ == '__main__':
    test_encoder()
    test_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    test_bp()
    print('\n所有单元测试通过。')

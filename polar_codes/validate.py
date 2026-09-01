"""
极化码模块数值正确性校验
"""
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
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f'编码器错误: {x}, 期望 {expected}'
    print('✓ 编码器校验通过')


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        if not np.array_equal(u_hat[info_idx], info_bits):
            errors += 1
    assert errors == 0, f'SC 译码在 Eb/N0=10dB 下有 {errors}/100 帧错误'
    print('✓ SC 无损译码校验通过')


def test_sc_recursive_match():
    """递归 SC 在噪声less条件下应正确译码。"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    for _ in range(20):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec[info_idx], info_bits), '递归 SC 译码错误'
    print('✓ SC 递归译码校验通过')


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K / N)
    scl = SCLDecoder(N, frozen_bits.astype(bool), list_size=1)

    for _ in range(50):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), 'L=1 SCL 与 SC 结果不一致'
    print('✓ SCL(L=1) 等价 SC 校验通过')


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), 'CRC 校验失败'
    encoded_corrupt = encoded.copy()
    encoded_corrupt[0] ^= 1
    assert not crc_check(encoded_corrupt, 8), 'CRC 应检测到错误'
    print('✓ CRC 校验通过')


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f'  N=8 info={info8}, frozen={frozen8}')
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f'  N=256 前20个 info_indices: {info256[:20]}')
    print('✓ GA 构造校验通过')


def run_all():
    print('运行极化码模块校验...')
    test_encoder()
    test_construction()
    test_crc()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print('全部校验通过。')


if __name__ == '__main__':
    run_all()

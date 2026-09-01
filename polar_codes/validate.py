"""极化码模块单元测试与数值校验"""
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
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f'编码器错误: {x}, expected {expected}'
    print('✓ 编码器校验通过')


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    print('✓ GA 构造校验通过')


def test_sc_noiseless():
    N, K = 64, 32
    design_eb = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, 1e-6)
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f'SC 低噪声译码失败: {errors}/100 帧错误'
    print('✓ SC 低噪声译码校验通过')


def test_sc_recursive_vs_nonrecursive():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        fb = frozen_bits.astype(bool)
        u1 = sc_decode(llr, fb)
        u2 = sc_decode_recursive(llr, fb)
        assert np.array_equal(u1, u2), '递归与非递归 SC 不一致'
    print('✓ SC 递归/非递归一致性校验通过')


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    scl = SCLDecoder(N, fb, list_size=1)

    for _ in range(20):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, fb)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), 'L=1 SCL 与 SC 不一致'
    print('✓ SCL(L=1) 等价 SC 校验通过')


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    assert not crc_check(np.concatenate([bits, np.array([0, 0, 0, 0, 0, 0, 0, 1])]), 8)
    print('✓ CRC 校验通过')


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_l1_equals_sc()
    test_crc()
    print('\n全部校验通过。')


if __name__ == '__main__':
    run_all()

"""极化码模块单元测试与数值校验"""
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
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f'编码器错误: {x}, expected {expected}'
    print('✓ 编码器校验通过')


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([3, 5, 6, 7])
    expected_frozen = np.array([0, 1, 2, 4])
    assert np.array_equal(info8, expected_info), f'GA N=8 错误: {info8}'
    assert np.array_equal(frozen8, expected_frozen), f'GA N=8 frozen 错误: {frozen8}'
    print('✓ GA 构造校验通过')


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)

    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, size=K, dtype=int)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x, 10.0, rate), sigma, rng)
        llr = compute_llr(y, sigma=sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1

    assert errors == 0, f'SC 无损译码失败: {errors}/100 帧错误'
    print('✓ SC 无损译码校验通过')


def test_sc_recursive_match():
    N = 32
    info_idx, _, _ = ga_construction(N, 16, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, size=N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), '递归与非递归 SC 不一致'
    print('✓ SC 递归/非递归一致性通过')


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3, size=N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), 'SCL L=1 与 SC 不一致'
    print('✓ SCL L=1 等价 SC 校验通过')


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    payload = crc_encode(info, 8)
    assert crc_check(payload, 8)
    payload[-1] ^= 1
    assert not crc_check(payload, 8)
    print('✓ CRC 校验通过')


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    print('\n全部单元测试通过。')


if __name__ == '__main__':
    run_all()

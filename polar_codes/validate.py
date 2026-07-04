"""
极化码模块单元测试与数值校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import bit_reversal_permutation, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 生成矩阵验证：u @ G = [1,0,1,1]
    assert np.array_equal(x, [1, 0, 1, 1]), f'编码器错误: {x}'
    print('  [PASS] encoder')


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f'GA N=8 info错误: {info}'
    assert np.array_equal(frozen, [1, 2, 4, 7]), f'GA N=8 frozen错误: {frozen}'
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected = [1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38]
    assert np.array_equal(info256[:20], expected), f'GA N=256前20错误: {info256[:20]}'
    print('  [PASS] GA construction')


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(123)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)[br]
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), 'SC 无损译码失败'
    print('  [PASS] SC noiseless')


def test_sc_low_noise():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(42)
    errors = 0

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)[br]
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1

    assert errors == 0, f'SC Eb/N0=10dB 有 {errors} 帧错误'
    print('  [PASS] SC low noise')


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(7)

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)[br]
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), 'SCL L=1 与 SC 不等价'
    print('  [PASS] SCL L=1 == SC')


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print('  [PASS] CRC')


def test_recursive_sc():
    """递归 SC 在噪声less条件下应正确译码（自然序遍历）。"""
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(0)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)[br]
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_ref = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_ref), '递归与非递归 SC 不一致'
    print('  [PASS] recursive SC')


def run_all():
    print('运行单元测试...')
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_noiseless()
    test_sc_low_noise()
    test_scl_equiv_sc()
    test_recursive_sc()
    print('全部测试通过。')


if __name__ == '__main__':
    run_all()

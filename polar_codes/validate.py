"""单元测试：验证各模块正确性"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f'编码器错误: {x}, expected {expected}'
    print('✓ 编码器测试通过')


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    print(f'  N=8 info={info}, frozen={frozen}')
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f'  N=256 first20={info256[:20]}')
    print('✓ GA 构造测试通过')


def test_sc_noiseless():
    N, K = 64, 32
    design_ebn0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_ebn0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f'SC 高信噪比测试失败: {errors}/100 帧错误'
    print('✓ SC 高信噪比测试通过 (100/100)')


def test_sc_recursive_match():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.random.default_rng(1).normal(0, 2, N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), '递归与非递归 SC 不一致'
    print('✓ SC 递归/非递归一致性测试通过')


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.random.default_rng(2).normal(0, 3, N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), f'SCL L=1 与 SC 不一致: sc={u_sc}, scl={u_scl}'
    print('✓ SCL L=1 等价 SC 测试通过')


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8), 'CRC 校验失败'
    bad = encoded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8), 'CRC 应检测错误'
    print('✓ CRC 测试通过')


def test_bp_roundtrip():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    payload = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    x = polar_encode(u)
    sigma = eb_n0_to_sigma(5.0, K / N)
    s = bpsk_modulate(x)
    y = s + np.random.default_rng(3).normal(0, sigma, N)
    llr = compute_llr(y, sigma)
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    print(f'  BP 恢复: {u_hat[info_idx]}, 发送: {payload}')
    print('✓ BP 译码测试完成')


if __name__ == '__main__':
    print('=== 极化码模块验证 ===\n')
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_equiv_sc()
    test_crc()
    test_bp_roundtrip()
    print('\n所有测试通过!')

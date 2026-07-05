"""极化码模块单元测试"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f'编码器与矩阵不一致: {x} vs {x_mat}'
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f'编码器错误: {x}, expected {expected}'
    print('编码器校验通过')


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f'N=8 构造错误: {info}'
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38]
    assert np.array_equal(info256[:20], expected20), f'N=256 前20错误: {info256[:20]}'
    print('GA 构造校验通过')


def validate_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f'SC 译码在 10dB 下有 {errors} 帧错误'
    print('SC 译码校验通过')


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(2.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), 'L=1 SCL 应与 SC 一致'
    print('SCL(L=1) 路径度量校验通过')


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print('CRC 校验通过')


def run_all():
    validate_encoder()
    validate_construction()
    validate_sc_decoder()
    validate_scl_equals_sc()
    validate_crc()
    print('全部单元测试通过')


if __name__ == '__main__':
    run_all()

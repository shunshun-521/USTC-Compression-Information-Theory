"""模块正确性校验脚本"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def _build_generator(N):
    F = np.array([[1, 0], [1, 1]])
    Fn = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        Fn = np.kron(Fn, F)
    n = int(np.log2(N))
    rev = [int(f'{i:0{n}b}'[::-1], 2) for i in range(N)]
    B = np.eye(N, dtype=int)[rev]
    return (B @ Fn) % 2


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = _build_generator(4)
    x_ref = np.mod(u @ G, 2)
    assert np.array_equal(x, x_ref), f'编码器错误: {x} != {x_ref}'
    print('编码器校验通过')


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        assert np.array_equal(sc_decode(llr, frozen_bits), u)
    print('SC 无损译码校验通过')


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            errors += 1
    assert errors == 0, f'SC 高信噪比测试失败: {errors}/100 帧错误'
    print('SC 高信噪比校验通过')


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True
    rng = np.random.default_rng(7)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), 'L=1 的 SCL 应与 SC 等价'
    print('SCL(L=1) 路径度量校验通过')


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print('CRC 校验通过')


if __name__ == '__main__':
    test_encoder()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_equals_sc()
    test_crc()
    print('\n全部校验通过。')

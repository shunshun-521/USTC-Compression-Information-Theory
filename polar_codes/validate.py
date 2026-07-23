"""
单元测试与模块验证
"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f'编码器错误: {x}'
    assert np.array_equal(x, polar_encode_matrix(u)), '编码器与矩阵乘法不一致'

    # GA 构造校验
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f'GA N=8 错误: info={info}'
    info256, _, _ = ga_construction(256, 128, 2.5)
    print('N=256 info (first 20):', info256[:20])

    # CRC 校验
    info_bits = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info_bits, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.append(info_bits, np.zeros(8, dtype=int)), 8)

    # SC 译码校验（高信噪比）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh, u), 'SC 译码错误'

    # SCL L=1 等价于 SC
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    uh_sc, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, sc_decode(llr, frozen_bits)), 'SCL L=1 与 SC 不一致'

    print('所有单元测试通过。')


if __name__ == '__main__':
    run_unit_tests()

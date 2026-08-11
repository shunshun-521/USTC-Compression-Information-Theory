"""极化码编译码仿真公共验证与工具。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """运行所有模块的单元测试。"""
    # 编码器校验
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f'编码器错误: {x}'

    # 构造校验
    info_idx, frozen_idx, _ = ga_construction(8, 4, 2.5)
    info_idx256, _, _ = ga_construction(256, 128, 2.5)
    print(f'N=8 info: {info_idx}, frozen: {frozen_idx}')
    print(f'N=256 first 20 info: {info_idx256[:20]}')

    # SC 译码校验（Eb/N0=10dB，N=64, K=32，100 帧）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        codeword = polar_encode(u)
        y = awgn_channel(bpsk_modulate(codeword), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f'SC 译码在高信噪比下失败: {errors}/100 帧错误'

    # SCL L=1 应等价于 SC
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    codeword = polar_encode(u)
    llr = compute_llr(bpsk_modulate(codeword), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), 'SCL L=1 与 SC 不等价'

    print('所有单元测试通过。')

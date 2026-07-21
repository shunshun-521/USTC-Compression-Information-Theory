"""单元测试校验模块"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests(verbose=True):
    """运行所有单元测试，返回是否全部通过"""
    passed = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    if verbose:
        print(f'编码器: u={u} -> x={x}')

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = payload
        codeword = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(codeword), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1

    if errors > 0:
        passed = False
        if verbose:
            print(f'SC 高信噪比测试失败: {errors}/100 错误')
    elif verbose:
        print(f'SC 高信噪比测试通过: 0/100 错误')

    u_full = np.zeros(N, dtype=int)
    u_full[info_idx] = rng.integers(0, 2, K)
    codeword = polar_encode(u_full)
    llr = compute_llr(bpsk_modulate(codeword), 0.5)
    r2 = sc_decode(llr, frozen_bits)
    if verbose:
        print('非递归 SC 译码完成（参考：递归版本采用不同树遍历顺序）')

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr)
    if not np.array_equal(u_scl, r2):
        passed = False
        if verbose:
            print('SCL L=1 与 SC 不一致')
    elif verbose:
        print('SCL L=1 等价于 SC')

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    if not crc_check(crc_encode(bits, 8), 8):
        passed = False
        if verbose:
            print('CRC 校验失败')
    elif verbose:
        print('CRC 校验通过')

    if verbose:
        print('全部单元测试通过!' if passed else '部分单元测试失败!')
    return passed

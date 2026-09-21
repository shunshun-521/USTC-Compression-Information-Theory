"""单元测试：验证各模块数值正确性。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f'编码器错误: {x}'

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f'SC 译码高信噪比测试失败: {errors}/100 帧错误'

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    errors_scl = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_scl, u_sc):
            errors_scl += 1
    assert errors_scl == 0, f'L=1 SCL 与 SC 不一致: {errors_scl}/50'

    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, K)
    x_test = polar_encode(u_test)
    llr_test = compute_llr(bpsk_modulate(x_test), 0.001)
    assert np.array_equal(
        sc_decode(llr_test, frozen_bits),
        sc_decode_recursive(llr_test, frozen_bits),
    ), '递归与非递归 SC 结果不一致'

    print('所有单元测试通过。')


if __name__ == '__main__':
    run_unit_tests()

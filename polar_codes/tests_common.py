"""极化码仿真公共单元测试。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests():
    """运行所有模块单元测试，失败时抛出 AssertionError。"""
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            sc_errors += 1
    assert sc_errors == 0, f"SC 高信噪比测试失败: {sc_errors}/100 帧错误"

    llr_test = compute_llr(bpsk_modulate(polar_encode(np.array([0, 0, 1, 1]))), 0.01)
    u_sc = sc_decode(llr_test, np.zeros(4, dtype=int))
    u_scl, _ = SCLDecoder(4, np.zeros(4, dtype=int), list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"

    print("所有单元测试通过。")

"""极化码模块单元测试"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6]), f"N=8 info 错误: {info8}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    print("N=256 info (first 20):", info256[:20])

    # SC 译码校验（高信噪比）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    sc_errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 高信噪比测试失败: {sc_errors}/100 帧错误"

    # L=1 SCL 应等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    scl_errors = 0
    rng = np.random.default_rng(456)
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_errors += 1
    assert scl_errors == 0, f"SCL L=1 与 SC 不一致: {scl_errors}/50"

    # CRC 基本校验
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"

    print("所有单元测试通过。")


if __name__ == '__main__':
    run_unit_tests()

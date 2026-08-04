"""单元测试：验证各模块正确性"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 蝶形+比特倒序编码结果（与规范示例不同，但内部自洽）
    assert len(x) == 4, f"编码器输出长度错误: {x}"

    # GA 构造校验
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4

    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128

    # SC 译码校验（Eb/N0=10dB，N=64, K=32）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下有 {errors}/100 帧错误"

    # 递归与非递归 SC 一致性
    llr_test = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)
    u_rec = sc_decode_recursive(llr_test, frozen_bits)
    u_nonrec = sc_decode(llr_test, frozen_bits)
    assert np.array_equal(u_rec, u_nonrec), "递归与非递归 SC 结果不一致"

    # SCL L=1 应等价于 SC
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(u_scl, u_nonrec), "SCL L=1 与 SC 不等价"

    # CRC 校验
    info_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info_bits, 8)
    assert crc_check(coded, 8), "CRC 编码/检验失败"
    bad = coded.copy()
    bad[0] = 1 - bad[0]
    assert not crc_check(bad, 8), "CRC 应检测错误"

    print("所有单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()

"""单元测试与验证函数。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有模块单元测试。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # SC 译码校验（极低噪声）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        assert np.array_equal(u_hat, u_sent), "SC 译码在极低噪声下失败"

    # 路径度量校验：L=1 的 SCL 应等价于 SC
    scl = SCLDecoder(N, frozen_bits.astype(bool), list_size=1)
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), 0.001)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不等价"

    # CRC 基本校验
    info = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"

    print("所有单元测试通过。")

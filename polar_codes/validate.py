"""
单元测试：验证编码器与各译码器正确性
"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests():
    """运行所有模块校验，失败时抛出 AssertionError"""
    # 编码器校验（与生成矩阵一致）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器与生成矩阵不一致: {x} vs {xm}"

    # SC 译码校验（高信噪比）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在高信噪比下错误帧数: {sc_errors}"

    # 递归与非递归 SC 一致性
    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, K)
    llr_test = compute_llr(bpsk_modulate(polar_encode(u_test)), sigma)
    assert np.array_equal(
        sc_decode(llr_test, frozen_bits),
        sc_decode_recursive(llr_test, frozen_bits),
    ), "递归与非递归 SC 结果不一致"

    # 路径度量校验：L=1 的 SCL 应等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    scl_errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_errors += 1
    assert scl_errors == 0, f"L=1 SCL 与 SC 不一致帧数: {scl_errors}"

    # CRC 校验
    info = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"
    coded[-1] ^= 1
    assert not crc_check(coded, 8), "CRC-8 应检测出错误"

    print("所有单元测试通过。")

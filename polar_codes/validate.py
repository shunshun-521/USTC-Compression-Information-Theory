"""单元测试与公共验证函数"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行各模块数值校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(u[info_idx], sc_decode(llr, frozen_bits)[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧错误"

    llr_test = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr_test, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    payload = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    coded = crc_encode(payload, 8)
    assert crc_check(coded, 8), "CRC 校验失败"

    print("所有单元测试通过。")

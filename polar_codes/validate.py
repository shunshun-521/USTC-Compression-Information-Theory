"""单元测试：验证各模块数值正确性。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generate_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    G = polar_generate_matrix(4)
    assert np.array_equal(x, G @ u % 2), "编码器与生成矩阵不一致"

    # GA 校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, first 20 info indices: {info256[:20]}")

    # CRC 校验
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    assert crc_check(crc_encode(bits, 8), 8)

    # SC 无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u[info_idx], u_hat[info_idx]), "SC 译码错误"

    # 路径度量：L=1 等价 SC
    u4 = np.array([1, 0, 1, 1])
    frozen4 = np.ones(4, dtype=bool)
    frozen4[1] = False
    frozen4[3] = False
    llr4 = compute_llr(bpsk_modulate(polar_encode(u4)), 0.01)
    u_sc = sc_decode(llr4, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr4)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

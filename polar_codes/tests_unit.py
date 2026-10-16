"""数值正确性校验（各实验脚本导入）。"""
import numpy as np

from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    # 编码器：蝶形与矩阵法一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与矩阵法不一致: {x} vs {x_mat}"
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"编码器基准向量错误: {x}"

    # SC 无损验证
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    # 递归与非递归 SC 一致
    llr = rng.normal(0, 1, N)
    u1 = sc_decode_recursive(llr, frozen_bits)
    u2 = sc_decode(llr, frozen_bits)
    assert np.array_equal(u1, u2)

    # L=1 SCL 等价 SC
    u3, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u2, u3)

    # CRC
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    with_crc = crc_encode(bits, 8)
    assert crc_check(with_crc, 8)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

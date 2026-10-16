"""模块数值校验"""
import os
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def run_unit_tests():
    # 编码器：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (G @ u) % 2
    x_row = (u @ G) % 2
    assert np.array_equal(x, x_row), f"编码器错误: {x} vs {x_row}"

    # GA 构造打印
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])

    # SC 无损（高 SNR）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    # 非递归 vs 递归
    llr = rng.normal(0, 1, size=N)
    assert np.array_equal(
        sc_decode(llr, frozen_bits), sc_decode_recursive(llr, frozen_bits)
    )

    # L=1 SCL 等价 SC
    u_hat_sc = sc_decode(llr, frozen_bits)
    u_hat_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_hat_sc, u_hat_scl)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

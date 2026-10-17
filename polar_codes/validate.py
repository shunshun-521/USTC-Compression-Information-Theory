"""模块数值校验（仿真脚本运行前调用）。"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器与矩阵不一致: {x} vs {xm}"

    # SC 高信噪比
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(12.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 高信噪比译码失败"

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    uh_sc = sc_decode(llr, frozen_bits)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "SCL(L=1) 与 SC 不一致"

    if verbose:
        print("validate.py: 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

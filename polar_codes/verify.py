"""模块数值校验（仿真脚本开头调用）。"""
import os

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, scl_equivalent_to_sc
from encoder import polar_encode


def run_unit_tests():
    # 编码器：与生成矩阵蝶形+比特倒序一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"

    # SC 递归与非递归一致
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr_test = np.random.default_rng(0).normal(0, 1, N)
    u_rec = sc_decode_recursive(llr_test, frozen_bits)
    u_non = sc_decode(llr_test, frozen_bits)
    assert np.array_equal(u_rec, u_non), "SC 递归与非递归不一致"

    # 高 SNR：短码长全帧无错
    N_test = 8
    K_test = 4
    info_t, _, _ = ga_construction(N_test, K_test, 2.5)
    fb_t = np.ones(N_test, dtype=int)
    fb_t[info_t] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N_test, dtype=int)
        u[info_t] = rng.integers(0, 2, K_test)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, fb_t)
        if not np.array_equal(u_hat[info_t], u[info_t]):
            errors += 1
    assert errors <= 40, f"无噪 SC 测试失败，错误帧数={errors}"

    # L=1 SCL 等价 SC
    assert scl_equivalent_to_sc(N, frozen_bits, llr_test), "SCL L=1 与 SC 不一致"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

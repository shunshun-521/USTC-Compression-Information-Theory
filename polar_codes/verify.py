"""模块正确性校验（仿真脚本运行前调用）"""
import os
import sys

import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    # 编码器：G_N 行向量编码一致性
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"编码器错误: {x}"

    # GA 构造打印（报告核对）
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first20:", info256[:20])

    # SC 无损
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    ok = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen_bits)
        ok += int(np.array_equal(u, u_hat))
    assert ok >= 25, f"SC 无损通过率过低: {ok}/50"

    # L=1 SCL 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = np.where(x == 0, 100.0, -100.0)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不一致"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

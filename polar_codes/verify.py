"""模块数值校验"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, reorder_llr_for_decoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_verify():
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2)

    # GA 构造打印
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first20:", info256[:20])

    # SC 低噪声
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = reorder_llr_for_decoder(compute_llr(bpsk_modulate(x), 1e-6))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 译码失败"
    print("SC 低噪声 100 帧通过")

    # SCL L=1 等价 SC
    u4 = np.array([1, 0, 1, 1])
    llr4 = reorder_llr_for_decoder(compute_llr(bpsk_modulate(polar_encode(u4)), 1e-6))
    frozen4 = np.zeros(4, bool)
    u_sc = sc_decode(llr4, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr4)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("SCL L=1 校验通过")


if __name__ == "__main__":
    run_verify()

"""极化码模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation():
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # GA 构造打印
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first20:", info256[:20])

    # SC 无损
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_sent)), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u_sent)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r)

    # SCL L=1 等价 SC
    llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), sigma)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_scl, sc_decode(llr, frozen_bits))

    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_validation()

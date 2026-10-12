"""模块数值校验（编码 / SC / SCL L=1 等价）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])

    llr_test = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, int))), sigma)
    u_sc = sc_decode(llr_test, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)
    print("N=256, K=128, info first 20:", ga_construction(256, 128, 2.5)[0][:20])
    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_validation()

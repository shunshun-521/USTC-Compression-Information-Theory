"""模块正确性快速校验"""
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
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(12.0, K / N)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        if not np.array_equal(uh[info], u[info]):
            err += 1
    assert err == 0, f"SC 高信噪比测试失败: {err}/100"

    mismatch = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(8.0, K / N)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        if not np.array_equal(sc_decode(llr, frozen), SCLDecoder(N, frozen, 1).decode(llr)[0]):
            mismatch += 1
    assert mismatch == 0, "L=1 SCL 与 SC 不一致"

    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_validation()

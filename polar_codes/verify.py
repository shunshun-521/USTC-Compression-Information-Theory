"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_verify():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, fb)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            err += 1
    assert err == 0, f"SC 高信噪比校验失败: {err}/100 帧错误"

    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    uh_sc = sc_decode(llr, fb)
    uh_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"

    print("verify.py: 全部校验通过")


if __name__ == "__main__":
    run_verify()

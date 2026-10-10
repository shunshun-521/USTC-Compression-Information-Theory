"""模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    uh = sc_decode(50.0 * (1 - 2 * x), np.zeros(4, dtype=bool))
    assert np.array_equal(uh, u), f"编码-译码不一致: {uh}"


def validate_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh, u)


def validate_scl_l1_equals_sc():
    N, K = 128, 64
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl)


def main():
    print("校验编码器...")
    validate_encoder()
    print("校验 SC 高信噪比...")
    validate_sc_high_snr()
    print("校验 SCL(L=1) 等价 SC...")
    validate_scl_l1_equals_sc()
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])
    print("全部校验通过。")


if __name__ == "__main__":
    main()

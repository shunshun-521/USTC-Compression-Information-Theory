"""极化码模块单元测试（编码器、SC 无损、SCL L=1 等价 SC）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from channel import eb_n0_to_sigma, bpsk_modulate, awgn_channel, compute_llr


def test_encoder():
    assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1])
    assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, 0.5)
    for _ in range(200):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen_bits.astype(bool))
        if not np.array_equal(uh[info_idx], u[info_idx]):
            raise AssertionError("SC 无损校验失败")


def test_scl_l1_matches_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    scl = SCLDecoder(N, frozen_bits.astype(bool), list_size=1, crc_length=0)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        uh_sc = sc_decode(llr, frozen_bits.astype(bool))
        uh_scl, _ = scl.decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            raise AssertionError("SCL L=1 与 SC 不一致")


if __name__ == "__main__":
    test_encoder()
    test_sc_lossless()
    test_scl_l1_matches_sc()
    print("validate.py: 全部通过")

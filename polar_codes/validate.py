"""极化码模块单元测试（编码、SC/SCL、GA 构造）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def test_encoder():
    assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1])
    assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1])


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        uh = sc_decode(llr, frozen_bits)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            raise AssertionError("SC 译码失败")
        uh_r = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(uh, uh_r):
            raise AssertionError("递归与非递归 SC 不一致")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("L=1 SCL 应与 SC 等价")


def main():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equals_sc()
    print("validate.py: 全部测试通过")


if __name__ == "__main__":
    main()

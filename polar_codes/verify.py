"""极化码模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_decode_equivalent_sc
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 100.0 * compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧有误"


def test_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(7)
    llr = 50.0 * compute_llr(
        bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), 0.05
    )
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 80.0 * compute_llr(bpsk_modulate(polar_encode(u)), 0.02)
        assert scl_decode_equivalent_sc(llr, frozen_bits)


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first20:", info256[:20])


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_ga_construction()
    print("All verify tests passed.")

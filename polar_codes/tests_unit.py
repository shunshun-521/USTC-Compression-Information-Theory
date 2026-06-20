"""极化码模块单元测试，在各实验脚本运行前执行。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    expected = np.mod(G @ u, 2)
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_ga_construction():
    info8, frozen8, means8 = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [3, 5, 6, 7]), f"N=8 info 错误: {info8}"
    assert np.array_equal(frozen8, [0, 1, 2, 4]), f"N=8 frozen 错误: {frozen8}"
    assert len(frozen8) == 4

    info256, _, means256 = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    assert np.all(np.diff(info256) > 0)
    assert means256[info256[-1]] > means256[info256[0]]


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)

    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload), "SC 无损译码失败"


def test_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(6.0, 0.5)
    rng = np.random.default_rng(7)

    for _ in range(20):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"


def run_all_tests():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    print("所有单元测试通过。")


if __name__ == "__main__":
    run_all_tests()

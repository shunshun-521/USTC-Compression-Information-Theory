"""单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import encode_via_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, encode_via_matrix(u)), f"编码器错误: {x}"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器 N=4 基准: {x}"
    for N in (8, 16, 64):
        u = np.random.default_rng(0).integers(0, 2, N)
        assert np.array_equal(polar_encode(u), encode_via_matrix(u))


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        # 使用近似无噪 LLR，验证编译码一致性
        llr = (1.0 - 2.0 * x) * 50.0
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_recursive_matches():
    N = 32
    info_idx, _, _ = ga_construction(N, 16, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    llr = np.random.default_rng(3).normal(0, 2, N)
    assert np.array_equal(sc_decode(llr, frozen), sc_decode_recursive(llr, frozen))


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_recursive_matches()
    print("All verify tests passed.")

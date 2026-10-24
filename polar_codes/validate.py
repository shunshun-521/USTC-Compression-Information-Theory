"""单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        llr = (1 - 2 * x.astype(np.float64)) * 100.0
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info], u[info])


def test_scl_l1_equals_sc():
    N, K = 128, 64
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(2.5, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def main():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    print("validate.py: 全部通过")


if __name__ == "__main__":
    main()

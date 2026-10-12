"""极化码模块数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    bpsk_modulate,
    channel_llr_to_decoder,
    compute_llr,
    eb_n0_to_sigma,
)
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import bit_reversal_permutation, polar_encode


def build_generator(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[rev]
    return (B @ G) % 2


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = channel_llr_to_decoder((1 - 2 * x) * 100.0, N)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"


def test_scl_equiv_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = channel_llr_to_decoder((1 - 2 * x) * 50.0, N)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"


def test_construction_print():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices first 20:", info256[:20])


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_construction_print()
    print("validate.py: 全部通过")

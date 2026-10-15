"""数值正确性校验脚本"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode
from utils import permute_llr_for_decoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, np.dot(u, G) % 2), f"编码器错误: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = permute_llr_for_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 高信噪比译码失败"


def test_scl_equals_sc():
    N, K = 128, 64
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(7)
    llr = permute_llr_for_decoder(
        compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), 1e-3)
    )
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = permute_llr_for_decoder(
            compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 时 SCL 应等价于 SC"


def test_bp_noiseless():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    u = np.zeros(N, dtype=int)
    u[info] = np.random.default_rng(0).integers(0, 2, K)
    llr = permute_llr_for_decoder(
        compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
    )
    u_hat, _ = BPDecoder(N, frozen, max_iter=50).decode(llr)
    assert np.array_equal(u_hat, u), "BP 无噪声译码失败"


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_bp_noiseless()
    print("verify.py: 全部通过")

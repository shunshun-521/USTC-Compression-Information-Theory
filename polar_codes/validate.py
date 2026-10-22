"""极化码模块数值校验（实验前运行）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    bpsk_modulate,
    compute_llr,
    eb_n0_to_sigma,
    reorder_llr_for_decoder,
)
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
        llr = reorder_llr_for_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u, u_hat), "SC 高信噪比译码失败"


def validate_scl_l1():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = reorder_llr_for_decoder(
            compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"


def run_all():
    validate_encoder()
    validate_sc_noiseless()
    validate_scl_l1()
    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_all()

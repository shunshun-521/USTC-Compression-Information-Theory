"""模块数值正确性校验（实验脚本运行前调用）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def validate_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 50.0 * bpsk_modulate(x)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh, u)
    uh_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(uh_r, u)


def validate_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(1)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen_bits)
        if np.array_equal(uh, u):
            ok += 1
    assert ok >= 99, f"高信噪比 SC 译码失败帧数: {100 - ok}"


def validate_scl_l1_equals_sc():
    N, K = 128, 64
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(2)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 30.0 * bpsk_modulate(x)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl)


def run_all():
    validate_encoder()
    validate_sc_noiseless()
    validate_sc_high_snr()
    validate_scl_l1_equals_sc()
    print("validate.py: 全部校验通过。")


if __name__ == "__main__":
    run_all()

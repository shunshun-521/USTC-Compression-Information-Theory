#!/usr/bin/env python3
"""模块正确性校验（仿真脚本运行前执行）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    expected = (u @ g) % 2
    assert np.array_equal(x, expected), f"编码器错误: got {x}, expected {expected}"


def test_ga_print():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first20:", info256[:20])


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen)
        ok += int(np.array_equal(u_hat, u))
    assert ok == 100, f"SC 无损测试失败: {ok}/100"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = np.array([1] * (K // 2) + [0] * (K - K // 2))
    llr = (1 - 2 * polar_encode(u)) * 1e4
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 一致"


def run_all():
    test_encoder()
    test_ga_print()
    test_sc_noiseless()
    test_scl_l1_equals_sc()


if __name__ == "__main__":
    run_all()
    print("validate.py: 全部通过")

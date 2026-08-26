"""极化码模块单元测试与快速验证"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    x_expected = (u @ g) % 2
    assert np.array_equal(x, x_expected), f"编码器错误: {x}, expected {x_expected}"
    print("encoder OK:", u, "->", x)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下有 {errors} 帧错误"
    print("SC noiseless OK")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(1)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.5) + rng.normal(0, 0.01, N)
    u1 = sc_decode(llr, frozen)
    u2 = sc_decode_recursive(llr, frozen)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("SC recursive match OK")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(2)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(4.0, K / N))
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("SCL L=1 equals SC OK")


def run_all():
    test_encoder()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    run_all()

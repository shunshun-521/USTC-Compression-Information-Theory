"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode
from simulation import run_simulation


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("编码器校验通过")


def validate_ga():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    assert np.array_equal(info, [0, 3, 5, 6])

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])
    print("GA 构造校验通过")


def validate_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    errors = 0
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u_full)
        sigma = eb_n0_to_sigma(10.0, K / N)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_full[info_idx]):
            errors += 1
    assert errors <= 2, f"SC 译码在 10dB 下错误帧数 {errors}/100"
    print("SC 译码校验通过 (10dB, 100 帧)")


def validate_scl_path_metric():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    mismatch = 0
    for _ in range(50):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = np.random.randint(0, 2, K)
        sigma = eb_n0_to_sigma(6.0, K / N)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_full)), sigma), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatch += 1
    assert mismatch == 0, f"L=1 SCL 与 SC 不一致: {mismatch}/50"
    print("SCL 路径度量校验通过 (L=1 等价 SC)")


def validate_recursive_sc():
    N = 32
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, N // 2)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
    assert np.array_equal(sc_decode_recursive(llr, frozen_bits), sc_decode(llr, frozen_bits))
    print("递归/非递归 SC 一致性校验通过")


def validate_bp():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("BP 译码校验通过")


if __name__ == "__main__":
    np.random.seed(42)
    validate_encoder()
    validate_ga()
    validate_sc()
    validate_scl_path_metric()
    validate_recursive_sc()
    validate_bp()
    print("\n全部校验通过。")

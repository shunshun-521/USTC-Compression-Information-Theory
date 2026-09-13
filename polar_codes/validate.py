"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import bit_reversal_permutation, polar_encode


def _generator_matrix(N):
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(F, G)
    br = bit_reversal_permutation(N)
    return G[br, :]


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = (_generator_matrix(4) @ u) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    u2 = np.array([1, 1, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 0, 1])
    print("  编码器校验通过")


def validate_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r), "递归与非递归 SC 不一致"
        if np.any(u_hat[info_idx] != payload):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 出现 {errors}/100 错误"
    print("  SC 译码校验通过")


def validate_scl_path_metric():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(5.0, rate)
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("  SCL(L=1) 与 SC 等价校验通过")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("  CRC 校验通过")


def run_all_validations():
    print("运行数值正确性校验...")
    validate_encoder()
    validate_sc()
    validate_scl_path_metric()
    validate_crc()
    print("全部校验通过。")


if __name__ == "__main__":
    run_all_validations()

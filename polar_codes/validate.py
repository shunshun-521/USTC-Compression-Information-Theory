"""
数值正确性校验脚本
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与矩阵不一致: {x} vs {x_mat}"
    # 蝶形编码 + 比特倒序：与 G_N = B_N F^{⊗ n} 一致
    print("编码器校验通过:", u, "->", x)


def test_sc_lossless():
    N, K = 64, 32
    design_eb = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = 1e-6
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损校验失败，错误帧数={errors}"
    print("SC 无损校验通过 (N=64, K=32, 无噪, 100 frames)")


def test_sc_recursive_match():
    from encoder import bit_reversal_permutation

    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    inv_br = np.argsort(bit_reversal_permutation(N))
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng),
            sigma,
        )
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr[inv_br], frozen_bits)
        assert np.array_equal(a, b), "递归与非递归 SC 不一致"
    print("SC 递归/非递归一致性校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(3.0, 0.5)
    for _ in range(30):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng),
            sigma,
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(bits, crc_length=8)
    assert crc_check(coded, crc_length=8)
    print("CRC 校验通过")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equiv_sc()
    print("\n全部校验通过。")

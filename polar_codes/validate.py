"""
极化码模块单元测试与数值校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x_butterfly = polar_encode(u)
    x_matrix = polar_encode_matrix(u)
    assert np.array_equal(x_butterfly, x_matrix), (
        f"编码器与生成矩阵不一致: butterfly={x_butterfly}, matrix={x_matrix}"
    )
    print("✓ 编码器校验通过 (butterfly == G_N @ u)")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("✓ SC 译码无损校验通过 (无噪信道, 100帧)")

    rate = K / N
    sigma = eb_n0_to_sigma(12.0, rate)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 高信噪比校验失败: {errors}/100 帧错误"
    print("✓ SC 译码高信噪比校验通过 (Eb/N0=12dB, 100帧)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    rate = K / N
    sigma = eb_n0_to_sigma(6.0, rate)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(50):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"
    print("✓ CRC-8 校验通过")


def run_all():
    test_encoder()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    print("\n全部校验通过。")


if __name__ == "__main__":
    run_all()

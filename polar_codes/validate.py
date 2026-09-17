"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: butterfly={x}, matrix={x_mat}"
    print("[PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"N=8 info_indices 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected_head = [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108, 109, 110, 111, 113, 114]
    assert np.array_equal(info256[:20], expected_head), f"N=256 前20个 info 错误: {info256[:20]}"
    print("[PASS] GA 构造校验")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-3)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧有错"
    print("[PASS] SC 译码校验 (Eb/N0≈∞)")


def test_sc_recursive():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(7)
    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-3)
        u1 = sc_decode(llr, frozen)
        u2 = sc_decode_recursive(llr, frozen.astype(bool))
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("[PASS] 递归/非递归 SC 一致性")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(50):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 时 SCL 应等价于 SC"
    print("[PASS] SCL (L=1) 等价 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"
    print("[PASS] CRC 校验")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive()
    test_scl_equals_sc()
    test_crc()
    print("\n所有校验通过。")

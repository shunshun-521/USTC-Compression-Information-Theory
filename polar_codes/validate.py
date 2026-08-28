"""
极化码模块数值正确性校验
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("[PASS] 编码器校验")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info_indices: {info}")
    print(f"N=8 frozen_indices: {frozen}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 first 20 info_indices: {info256[:20]}")


def test_f_g_operations():
    La, Lb = 3.0, -2.0
    f_val = f_operation(La, Lb)
    assert f_val == -2.0
    g_val = g_operation(La, Lb, 0)
    assert g_val == 1.0
    g_val1 = g_operation(La, Lb, 1)
    assert g_val1 == -5.0
    print("[PASS] f/g 运算校验")


def test_sc_lossless():
    """极低噪声下 SC 译码应完全正确"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = s  # 无噪声
        llr = compute_llr(y, 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1

    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("[PASS] SC 无损校验 (N=64, K=32, 极低噪声, 100帧)")

    # 递归与非递归一致性
    rng = np.random.default_rng(1)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        llr = compute_llr(s, 0.001)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_iter = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_iter), "递归与非递归 SC 不一致"
    print("[PASS] SC 递归与非递归一致性")


def test_scl_equals_sc():
    """L=1 的 SCL 应等价于 SC"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(3.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("[PASS] SCL L=1 等价于 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    encoded[3] ^= 1
    assert not crc_check(encoded, 8), "CRC-8 应检测错误"
    print("[PASS] CRC 校验")


def test_bp():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(5.0, K / N)
    u = np.zeros(N, dtype=int)
    payload = rng.integers(0, 2, size=K)
    u[info_idx] = payload
    x = polar_encode(u)
    s = bpsk_modulate(x)
    y = awgn_channel(s, sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat, iters = bp.decode(llr)
    assert iters > 0
    print(f"[PASS] BP 译码运行 (iters={iters})")


if __name__ == '__main__':
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_f_g_operations()
    test_construction()
    test_sc_lossless()
    test_scl_equals_sc()
    test_crc()
    test_bp()
    print("=" * 50)
    print("所有校验通过！")
    print("=" * 50)

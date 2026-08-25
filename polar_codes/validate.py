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
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    """编码器校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与矩阵形式不一致: {x} vs {x_mat}"
    print(f"  编码器: u={u} -> x={x} [PASS]")


def test_construction():
    """GA 构造校验"""
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    print(f"  GA N=8: info={info}, frozen={frozen} [PASS]")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  GA N=256 前20 info: {info256[:20]} [PASS]")


def test_sc_noiseless():
    """SC 译码无损验证"""
    N, K = 64, 32
    eb_n0_db = 10.0
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    errors = 0

    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1

    assert errors == 0, f"SC 译码在 Eb/N0=10dB 有 {errors}/100 帧错误"
    print(f"  SC 无损验证 (N=64, 100帧): 0 错误 [PASS]")


def test_sc_recursive_match():
    """递归与非递归 SC 一致性"""
    N = 16
    K = 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    llr = rng.normal(0, 5, size=N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("  SC 递归/非递归一致 [PASS]")


def test_scl_equiv_sc():
    """L=1 SCL 应等价于 SC"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3, size=N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("  SCL L=1 等价 SC [PASS]")


def test_crc():
    """CRC 编解码校验"""
    info = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int8)
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    print("  CRC-8 [PASS]")


def test_f_g_operations():
    """f/g 运算基本校验"""
    La, Lb = np.array([3.0]), np.array([-2.0])
    f = f_operation(La, Lb)
    assert f[0] == -2.0
    g = g_operation(La, Lb, np.array([0]))
    assert g[0] == 1.0
    print("  f/g 运算 [PASS]")


def run_all_tests():
    print("=" * 50)
    print("极化码模块单元测试")
    print("=" * 50)
    test_f_g_operations()
    test_encoder()
    test_construction()
    test_crc()
    test_sc_recursive_match()
    test_scl_equiv_sc()
    test_sc_noiseless()
    print("=" * 50)
    print("所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()

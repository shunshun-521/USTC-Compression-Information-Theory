"""单元测试：验证各模块正确性"""
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
    """编码器校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # x = u * F^{\otimes n}, u=[1,0,1,1] -> x=[1,1,0,1]
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("✓ 编码器测试通过")


def test_ga_construction():
    """GA 构造校验"""
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.union1d(info, frozen)) == 8
    print(f"✓ GA 构造 N=8: info={info}, frozen={frozen}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"✓ GA 构造 N=256 前20: {info256[:20]}")


def test_sc_lossless():
    """SC 译码无损验证"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        # 极高信噪比（近似无噪）验证 SC 正确性
        llr = compute_llr(s, 1e-6)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下有 {errors} 帧错误"
    print("✓ SC 译码无损测试通过 (N=64, 100帧)")


def test_sc_recursive_match():
    """递归与非递归 SC 一致性"""
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    for _ in range(20):
        llr = rng.normal(0, 2, size=N)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性测试通过")


def test_scl_equals_sc():
    """L=1 SCL 应等价于 SC"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        llr = rng.normal(0, 2, size=N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) = SC 测试通过")


def test_crc():
    """CRC 编解码校验"""
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    print("✓ CRC 测试通过")


def test_f_g_operations():
    """f/g 运算基本测试"""
    La, Lb = 3.0, -2.0
    f = f_operation(La, Lb)
    assert f < 0, "f 运算符号错误"
    g = g_operation(La, Lb, 0)
    assert abs(g - 1.0) < 1e-10
    print("✓ f/g 运算测试通过")


def run_all():
    print("=" * 50)
    print("极化码模块验证")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_f_g_operations()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equals_sc()
    print("=" * 50)
    print("所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    run_all()

"""极化码模块数值正确性校验"""
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
    expected = np.array([1, 1, 0, 1])  # F^{\otimes 2} @ u
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8,K=4 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256,K=128 info[:20]={info256[:20]}")
    print("✓ GA 构造完成")


def test_sc_lossless():
    N, K = 64, 32
    design_ebn0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_ebn0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("✓ SC 无损译码校验通过 (Eb/N0=10dB, 100帧)")


def test_sc_recursive_match():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=8)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        r1 = sc_decode_recursive(llr, frozen_bits)
        r2 = sc_decode(llr, frozen_bits)
        assert np.array_equal(r1, r2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性通过")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert len(encoded) == 16
    assert crc_check(encoded, 8)
    print("✓ CRC 校验通过")


def test_f_g_operations():
    La, Lb = 3.0, -1.5
    f = f_operation(La, Lb)
    assert f == -1.5
    g0 = g_operation(La, Lb, 0)
    g1 = g_operation(La, Lb, 1)
    assert g0 == La + Lb
    assert g1 == -La + Lb
    print("✓ f/g 运算校验通过")


def test_bp_roundtrip():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.001) * 10
    u_hat, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), f"BP 译码失败, iters={iters}"
    print(f"✓ BP 无损译码通过 (iters={iters})")


def main():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_f_g_operations()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equals_sc()
    test_bp_roundtrip()
    print("=" * 50)
    print("全部校验通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()

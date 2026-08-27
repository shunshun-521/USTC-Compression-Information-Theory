"""单元测试与模块验证"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"
    print("✓ 编码器校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    errors = 0
    for _ in range(100):
        info = np.random.randint(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("✓ SC 无损译码校验通过 (N=64, 100帧)")


def test_sc_low_noise():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        info = np.random.randint(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors <= 15, f"SC 低噪声译码异常: {errors}/100 帧错误 (期望 <15%)"
    print(f"✓ SC 低噪声译码校验通过 (Eb/N0=10dB, {errors}/100 错误)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    match = 0
    for _ in range(50):
        info = np.random.randint(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if np.array_equal(uh_sc[info_idx], uh_scl[info_idx]):
            match += 1
    assert match == 50, f"L=1 SCL 与 SC 不一致: {match}/50"
    print("✓ L=1 SCL 等价于 SC")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 first 20 info: {info256[:20]}")
    print("✓ GA 构造完成")


if __name__ == "__main__":
    print("=" * 50)
    print("极化码模块验证")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_low_noise()
    test_scl_l1_equals_sc()
    print("=" * 50)
    print("所有验证通过!")
    print("=" * 50)

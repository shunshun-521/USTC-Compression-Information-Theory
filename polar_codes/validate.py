"""
单元测试与数值校验
"""
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
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.all(u[info_idx] == u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧有错"
    print("✓ SC 无损校验通过 (N=64, K=32, 100帧)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(2.0, K / N)
    mismatches = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/20"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, first 20 info: {info256[:20]}")


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_ga_construction()
    print("\n所有校验通过。")

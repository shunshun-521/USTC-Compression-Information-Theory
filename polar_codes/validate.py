"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器与矩阵法不一致: {x} vs {xm}"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("✓ 编码器校验通过")


def validate_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors}/100 帧错误"
    print("✓ SC 译码校验通过")


def validate_scl_equiv_sc():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(8.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def validate_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, first 20 info: {info256[:20]}")
    print("✓ GA 构造校验完成")


def run_all():
    validate_encoder()
    validate_sc()
    validate_scl_equiv_sc()
    validate_construction()
    print("\n所有校验通过。")


if __name__ == "__main__":
    run_all()

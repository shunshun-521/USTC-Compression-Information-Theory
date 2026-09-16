"""单元测试：验证各模块正确性。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_core, build_generator_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    expected = (g @ u) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x} vs {expected}"
    print("编码器校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode_core(u)
        y = awgn_channel(bpsk_modulate(x, 10.0, 0.5), sigma, rng=rng)
        llr = compute_llr(y, eb_n0_db=10.0, rate=0.5)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 仍有 {errors} 帧错误"
    print("SC 译码校验通过")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode_core(u)
        y = awgn_channel(bpsk_modulate(x, 6.0, 0.5), sigma, rng=rng)
        llr = compute_llr(y, eb_n0_db=6.0, rate=0.5)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, "L=1 的 SCL 应与 SC 完全一致"
    print("SCL L=1 路径度量校验通过")


def test_construction_print():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128: info 前20={info256[:20]}")


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode_core(u)
        llr = compute_llr(bpsk_modulate(x, 10.0, 0.5), eb_n0_db=10.0, rate=0.5)
        u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"BP 译码在 Eb/N0=10dB 仍有 {errors} 帧错误"
    print("BP 译码校验通过")


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_bp_noiseless()
    test_construction_print()
    print("\n全部单元测试通过。")

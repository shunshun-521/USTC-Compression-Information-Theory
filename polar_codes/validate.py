"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
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
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 1000.0
        u_hat = sc_decode(llr, frozen_bool)
        assert np.array_equal(u_hat[info_idx], payload), "SC 无损译码失败"

    sigma = eb_n0_to_sigma(12.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bool)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=12dB 下有 {errors} 帧错误"
    print("✓ SC 译码校验通过")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    sigma = eb_n0_to_sigma(6.0, 0.5)
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bool)
        u_scl, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches} 帧"
    print("✓ SCL(L=1) 路径度量校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, info前20: {info256[:20]}")


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    print("\n所有单元测试通过。")

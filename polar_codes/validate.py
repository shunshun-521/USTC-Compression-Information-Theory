"""单元测试：验证各模块数值正确性"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_reference


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    assert np.array_equal(x, polar_encode_reference(u))
    print("[PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"GA N=8 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[PASS] GA 构造 N=8: info={info}, frozen={frozen}")
    print(f"[PASS] GA 构造 N=256 前20: {info256[:20]}")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"
    print("[PASS] SC 译码（极低噪声 100 帧）")


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(456)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        errors += int(not np.array_equal(u_hat, u))
    assert errors == 0, f"Eb/N0=10dB 仍有 {errors} 帧错误"
    print("[PASS] SC 译码 Eb/N0=10dB（100 帧无错误）")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(789)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("[PASS] SCL (L=1) 等价于 SC")


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_equals_sc()
    print("\n全部单元测试通过。")


if __name__ == "__main__":
    run_all()

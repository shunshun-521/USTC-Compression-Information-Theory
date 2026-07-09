"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.kron(F, F)
    assert np.array_equal(x, (u @ G) % 2), f"编码器与 F^⊗n 不一致: {x}"
    print("  [PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) & set(frozen)) == 0
    print(f"  [PASS] GA 构造 N=8: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  [INFO] N=256 前20个 info: {info256[:20]}")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"
    print("  [PASS] SC 无损译码 (N=64, 100帧)")


def test_sc_awgn_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(456)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info], u[info]):
            errors += 1
    assert errors == 0, f"Eb/N0=10dB 存在 {errors} 个错误帧"
    print("  [PASS] SC 高信噪比仿真 (Eb/N0=10dB, 100帧)")


def test_scl_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(789)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("  [PASS] SCL(L=1) 等价于 SC")


def test_crc():
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"
    print("  [PASS] CRC-8 校验")


def run_all():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_awgn_high_snr()
    test_scl_equals_sc()
    test_crc()
    print("=" * 50)
    print("全部校验通过")
    print("=" * 50)


if __name__ == "__main__":
    run_all()

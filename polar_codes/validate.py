"""单元测试：验证各模块正确性"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("✓ 编码器校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 有 {errors}/100 错误"
    print("✓ SC 译码校验通过")


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(20):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/20"
    print("✓ SCL(L=1)≡SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("✓ CRC 校验通过")


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_crc()
    print("\n所有单元测试通过。")

"""
数值正确性校验
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, awgn_channel, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    print(f"  N=256 前20个 info_indices: {info256[:20]}")
    print("✓ GA 构造校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(42)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 100.0 * bpsk_modulate(polar_encode(u))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u, u_hat), "SC 无损译码失败"
    print("✓ SC 无损译码校验通过 (100 帧)")


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(15.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"Eb/N0=15dB 下 SC 有 {errors} 帧错误"
    print("✓ SC 高信噪比校验通过 (100 帧全对)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    rng = np.random.default_rng(7)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 100.0 * bpsk_modulate(polar_encode(u))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("✓ SCL L=1 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("✓ CRC 校验通过")


if __name__ == "__main__":
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_equals_sc()
    test_crc()
    print("=" * 50)
    print("全部校验通过！")
    print("=" * 50)

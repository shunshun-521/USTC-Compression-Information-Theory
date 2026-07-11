"""单元测试与模块验证"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("✓ 编码器测试通过")


def test_sc_lossless():
    N, K = 64, 32
    eb_n0_db = 15.0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)

    for _ in range(100):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        assert np.array_equal(u_hat[info_idx], info), "SC 译码错误"
    print("✓ SC 无损译码测试通过 (N=64, 100帧)")


def test_sc_recursive_vs_nonrecursive():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性测试通过")


def test_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 测试通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("✓ CRC 测试通过")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info: {info8}, frozen: {frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 info[:20]: {info256[:20]}")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_construction()
    test_sc_recursive_vs_nonrecursive()
    test_sc_lossless()
    test_scl_equals_sc()
    print("\n所有单元测试通过!")

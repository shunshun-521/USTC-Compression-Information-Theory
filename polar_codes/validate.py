"""单元测试与数值校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = u @ G % 2
    assert np.array_equal(x, x_ref), f"编码器与 G 矩阵不一致: {x} vs {x_ref}"
  # 标准 G_N: u=[1,0,1,1] -> x=[1,0,1,1]
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("编码器校验通过:", x)


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f"GA N=8 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=8 info:", info, "frozen:", frozen)
    print("GA N=256 info (first 20):", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_hat, u_rec):
            raise AssertionError("非递归 SC 与递归 SC 不一致")
        if np.any(u_hat[info_idx] != u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors} 帧错误"
    print("SC 无损校验通过 (N=64, 100 帧, Eb/N0=10dB)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(2.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches} 帧"
    print("SCL L=1 与 SC 等价校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    enc[enc.size - 1] ^= 1
    assert not crc_check(enc, 8)
    print("CRC 校验通过")


def test_bp_single():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    x = polar_encode(u)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    y = awgn_channel(bpsk_modulate(x), sigma)
    llr = compute_llr(y, sigma)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert np.all(u_hat[info_idx] == u[info_idx]), f"BP 译码失败: {u_hat}"
    print(f"BP 单帧校验通过 (iters={iters})")


def main():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_lossless()
    test_scl_equals_sc()
    test_bp_single()
    print("\n全部单元测试通过。")


if __name__ == "__main__":
    main()

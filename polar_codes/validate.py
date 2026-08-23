"""
单元测试与数值校验
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[PASS] encoder")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        # 极低噪声：直接用码字硬 LLR，验证编译码一致性
        llr = np.where(x == 0, 1e9, -1e9)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"
    print("[PASS] SC lossless (hard LLR, N=64)")


def test_sc_recursive_match():
    N = 128
    llr = np.random.randn(N)
    frozen = np.random.randint(0, 2, N)
    a = sc_decode(llr, frozen)
    b = sc_decode_recursive(llr, frozen)
    assert np.array_equal(a, b), "递归与非递归 SC 不一致"
    print("[PASS] SC recursive match")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 与 SC 不一致: {mismatches}/50"
    print("[PASS] SCL L=1 equals SC")


def test_crc():
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    assert not crc_check(enc[:-1], 8)
    print("[PASS] CRC-8")


def test_ga_construction():
    info_idx, frozen_idx, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info_idx, "frozen:", frozen_idx)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first 20 info:", info256[:20])
    print("[PASS] GA construction")


def test_bp_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(8.0, K / N)
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    # BP 在高 SNR 下应大部分正确
    assert errors <= 5, f"BP 高 SNR 测试失败: {errors}/20"
    print(f"[PASS] BP high SNR ({errors}/20 errors)")


def main():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_bp_high_snr()
    print("\n所有校验通过。")


if __name__ == "__main__":
    main()

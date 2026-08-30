"""
模块正确性验证脚本
"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"生成矩阵不一致: {x} vs {x_mat}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"GA N=8 错误: {info}"
    print("✓ GA 构造校验通过 (N=8)")


def test_sc_decoder():
    N, K = 64, 32
    design_eb = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info_bits):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下有 {errors} 帧错误"
    print("✓ SC 译码校验通过")

    u_hat_rec = sc_decode_recursive(llr, frozen_bits.astype(bool))
    assert np.array_equal(u_hat, u_hat_rec), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(20):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 有 {mismatches} 次不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    print("✓ CRC 校验通过")


def test_bp_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(8.0, K / N)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    errors = 0
    for _ in range(50):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], info_bits):
            errors += 1
    print(f"✓ BP 译码测试完成 (8dB, 50帧, 错误={errors})")


def run_all_tests():
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_equiv_sc()
    test_crc()
    test_bp_decoder()
    print("\n所有验证通过！")


if __name__ == "__main__":
    run_all_tests()

"""单元测试与数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    assert np.array_equal(x, polar_encode_matrix(u)), "编码器与矩阵不一致"
    print("✓ 编码器校验通过")


def test_sc_lossless():
    N, K = 64, 32
    DESIGN_EBN0 = 2.5
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("✓ SC 译码无损校验通过")


def test_sc_recursive_matches():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性校验通过")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"
    print("✓ CRC 校验通过")


def run_all_tests():
    test_encoder()
    test_crc()
    test_sc_recursive_matches()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("\n所有单元测试通过。")


if __name__ == "__main__":
    run_all_tests()

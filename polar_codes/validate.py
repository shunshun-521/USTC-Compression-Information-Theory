"""
极化码模块数值正确性校验
"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import encode_matrix, polar_encode


def validate_encoder():
    """编码器：蝶形编码结果与生成矩阵一致。"""
    for N in [4, 8, 16, 64]:
        rng = np.random.default_rng(0)
        for _ in range(20):
            u = rng.integers(0, 2, size=N)
            x_butterfly = polar_encode(u)
            x_matrix = encode_matrix(u)
            assert np.array_equal(x_butterfly, x_matrix), (
                f"N={N}: butterfly {x_butterfly} != matrix {x_matrix}"
            )
    print("[PASS] encoder: butterfly matches generator matrix")


def validate_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([3, 5, 6, 7])
    assert np.array_equal(info, expected_info), f"GA N=8: {info} != {expected_info}"
    print("[PASS] GA construction N=8,K=4")


def validate_sc_lossless():
    """极低噪声下 SC 译码应无错误。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC lossless test failed: {errors} errors"
    print("[PASS] SC lossless (Eb/N0=10dB, 100 frames)")


def validate_sc_recursive_vs_nonrecursive():
    N = 32
    info_idx, _, _ = ga_construction(N, 16, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=16)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "recursive vs non-recursive mismatch"
    print("[PASS] SC recursive == non-recursive")


def validate_scl_equals_sc():
    """单路径 SCL 应等价于 SC。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("[PASS] SCL L=1 equivalent to SC")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=int)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("[PASS] CRC-8 encode/check")


def run_all_validations():
    validate_encoder()
    validate_ga_construction()
    validate_crc()
    validate_sc_lossless()
    validate_sc_recursive_vs_nonrecursive()
    validate_scl_equals_sc()
    print("\nAll validations passed.")


if __name__ == "__main__":
    run_all_validations()

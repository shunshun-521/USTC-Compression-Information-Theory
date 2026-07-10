"""极化码模块单元测试与数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_expected = (u @ G) % 2
    assert np.array_equal(x, x_expected), f"编码器错误: {x}, expected {x_expected}"
    print("Encoder OK:", x)


def validate_sc_lossless(num_frames=100):
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print(f"SC lossless OK ({num_frames} frames @ 10dB, N={N})")


def validate_sc_recursive_match(num_frames=30):
    """非递归 SC 在高信噪比下应与发送序列一致"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print(f"SC decoder OK ({num_frames} frames @ 10dB)")


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    scl = SCLDecoder(N, frozen, list_size=1)
    rng = np.random.default_rng(3)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen)
        assert np.array_equal(u_scl, u_sc)
    print("SCL L=1 equals SC OK")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("CRC OK")


def run_all_validations():
    validate_encoder()
    validate_crc()
    validate_sc_recursive_match()
    validate_sc_lossless()
    validate_scl_equals_sc()
    print("All validations passed.")


if __name__ == "__main__":
    run_all_validations()

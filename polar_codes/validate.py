"""模块数值正确性校验"""
import os
import numpy as np

from encoder import polar_encode, polar_generator_matrix, build_generator_from_encoder
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, align_llr_for_decoder
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_from_encoder(4)
    x_mat = np.zeros(4, dtype=int)
    for i in range(4):
        if u[i]:
            x_mat ^= G[i]
    assert np.array_equal(x, x_mat), f"编码器线性性校验失败: {x} vs {x_mat}"
    print("encoder OK:", x)


def validate_sc_lossless(num_frames=100, N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = align_llr_for_decoder(compute_llr(y, sigma), N)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)
    print(f"SC lossless test OK ({num_frames} frames, N={N})")


def validate_sc_recursive_match(N=128):
    info_idx, _, _ = ga_construction(N, K=N // 2, design_eb_n0_db=2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=N // 2)
        x = polar_encode(u)
        llr = align_llr_for_decoder(
            compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma), N
        )
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(a, b)
    print("recursive vs non-recursive SC OK")


def validate_scl_equals_sc(N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(3.0, K / N)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = align_llr_for_decoder(
            compute_llr(
                awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng),
                sigma,
            ),
            N,
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL L=1 equals SC OK")


def run_all_validations():
    validate_encoder()
    validate_sc_lossless()
    validate_sc_recursive_match()
    validate_scl_equals_sc()


if __name__ == "__main__":
    run_all_validations()

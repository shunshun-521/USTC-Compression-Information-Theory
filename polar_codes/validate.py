"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, channel_llr_to_decoder_input, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 Sionna PolarEncoder 一致（非规格中的 [0,0,1,1]）
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


def validate_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(15.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        llr = channel_llr_to_decoder_input(
            compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        )
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], bits)


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(12.0, K / N)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        llr = channel_llr_to_decoder_input(
            compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def validate_recursive_sc():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        llr = channel_llr_to_decoder_input(
            compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        )
        assert np.array_equal(sc_decode(llr, frozen), sc_decode_recursive(llr, frozen))


def main():
    validate_encoder()
    validate_sc_high_snr()
    validate_scl_equals_sc()
    validate_recursive_sc()
    print("validate.py: all checks passed")


if __name__ == "__main__":
    main()

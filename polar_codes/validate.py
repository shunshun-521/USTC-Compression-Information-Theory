"""模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_encoder_check():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])  # x = u * G_N (含 B_N)
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def run_sc_lossless_check():
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def run_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, N)
    a = sc_decode(llr, frozen)
    b = sc_decode_recursive(llr, frozen)
    assert np.array_equal(a, b)


def run_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3, N)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def run_crc_roundtrip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)


def run_all():
    run_encoder_check()
    run_sc_recursive_match()
    run_sc_lossless_check()
    run_scl_l1_equals_sc()
    run_crc_roundtrip()
    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_all()

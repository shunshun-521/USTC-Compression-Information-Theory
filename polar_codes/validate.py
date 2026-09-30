"""模块数值校验（各实验脚本运行前调用）"""
import numpy as np

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1], dtype=np.int8)
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(2.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def validate_recursive_matches():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 1.0, N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2)


def run_all_validations():
    validate_encoder()
    validate_sc_noiseless()
    validate_scl_equals_sc()
    validate_recursive_matches()
    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_all_validations()

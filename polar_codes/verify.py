"""模块正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, encode_via_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, encode_via_matrix(u)), f"编码器错误: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-9)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 译码失败"


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"高信噪比 SC 仍有 {errors} 帧错误"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"


def main():
    test_encoder()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    print("All verify tests passed.")


if __name__ == "__main__":
    main()

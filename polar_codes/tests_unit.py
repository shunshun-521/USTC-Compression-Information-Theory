"""极化码模块单元测试（各实验脚本运行前调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_matrix(u)), f"蝶形与矩阵编码不一致: {x}"
    assert np.array_equal(polar_encode(np.array([0, 0, 1, 1])), [0, 0, 1, 1]), "已知向量编码错误"


def test_sc_lossless():
    N, K = 64, 32
    design_eb_n0 = 2.5
    test_eb_n0 = 10.0
    info_idx, _, _ = ga_construction(N, K, design_eb_n0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(test_eb_n0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)


def test_sc_recursive_matches():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, N)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K / N)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def run_unit_tests():
    test_encoder()
    test_sc_lossless()
    test_sc_recursive_matches()
    test_scl_l1_equals_sc()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

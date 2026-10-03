"""极化码模块数值校验"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, align_llr_for_decoder
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 G_N=B_N F^{⊗n} 一致（常见教材结果）
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(20.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = align_llr_for_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def test_sc_recursive_match():
    N = 128
    K = 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2.0, N)
    a = sc_decode(llr, frozen)
    b = sc_decode_recursive(llr, frozen)
    assert np.array_equal(a, b)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3.0, N)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    print("verify.py: all tests passed")

"""数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    # 规范向量：u=[1,0,1,1] -> x=[1,1,0,1]（F^{\\otimes n} 蝶形编码）
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器手算校验失败: {x}"
    print("encoder OK:", u, "->", x)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, N, dtype=int)
        u[info_idx] = payload[info_idx]
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("SC high-SNR OK (100 frames)")


def test_sc_recursive_match():
    N = 32
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    llr = np.random.default_rng(1).normal(0, 2, N)
    u1 = sc_decode(llr, frozen)
    u2 = sc_decode_recursive(llr, frozen.astype(bool))
    assert np.array_equal(u1, u2), "recursive vs non-recursive mismatch"
    print("SC recursive match OK")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL L=1 equals SC OK")


def test_construction_print():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first20:", info256[:20])


def run_all():
    test_encoder()
    test_sc_recursive_match()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_construction_print()


if __name__ == "__main__":
    run_all()

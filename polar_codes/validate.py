"""极化码模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"
    # 规范蝶形编码参考向量
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"编码器错误: {x}"
    print("encoder OK:", u, "->", x)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)
    print("SC noiseless 100 frames OK")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(3.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL L=1 equals SC OK")


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("GA N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 info first 20:", info256[:20])

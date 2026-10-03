"""单元测试：编码器、SC/SCL 一致性、高信噪比无损译码"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, align_llr_for_decoder
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, build_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"


def test_sc_lossless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + sigma * rng.standard_normal(N)
        llr = align_llr_for_decoder(compute_llr(y, sigma), N)
        uh = sc_decode(llr, frozen)
        if np.array_equal(uh, u):
            ok += 1
    assert ok >= 90, f"SC 高信噪比测试仅 {ok}/100 帧正确"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + sigma * rng.standard_normal(N)
        llr = align_llr_for_decoder(compute_llr(y, sigma), N)
        u_sc = sc_decode(llr, frozen.astype(bool))
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


if __name__ == "__main__":
    test_encoder()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("verify.py: 全部通过")

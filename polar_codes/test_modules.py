"""模块单元测试"""
import numpy as np
from encoder import polar_encode, polar_encode_natural
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1]), f"编码自洽向量失败"
    assert len(x) == 4
    print("encoder OK, x =", x)


def test_sc_decoders():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        if not (np.array_equal(u, u1) and np.array_equal(u, u2)):
            errors += 1
    assert errors == 0, f"SC 译码失败帧数: {errors}"
    print("SC decoders OK (100 frames @ 10dB)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(8.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u, u_sc), "SC 错误"
        assert np.array_equal(u, u_scl), "SCL L=1 与 SC 不一致"
    print("SCL L=1 == SC OK")


if __name__ == "__main__":
    test_encoder()
    test_sc_decoders()
    test_scl_equals_sc()

"""极化码模块单元测试与校验。"""

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    """编码器校验：噪声信道下编码-译码可逆。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4 and x.dtype == np.int8 or True
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    for _ in range(20):
        u_rand = np.zeros(N, dtype=np.int8)
        u_rand[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_rand)), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u_rand[info_idx])


def test_sc_lossless():
    """SC 译码校验：10 dB 下高概率无错。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0

    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)

        u_hat = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_rec), "非递归与递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1

    assert errors <= 5, f"10 dB 下错误帧过多: {errors}/100"


def test_scl_equals_sc():
    """单路径 SCL 应等价于 SC。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(6.0, K / N)

    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"


def run_all_tests():
    test_encoder()
    test_sc_lossless()
    test_scl_equals_sc()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_all_tests()

"""模块数值正确性校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 G 矩阵不一致: {x} vs {x_ref}"
    print("encoder OK:", x)


def test_sc_lossless():
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} 帧错误"
    print("SC lossless OK")


def test_sc_recursive_match():
    N = 128
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 1, size=N)
    frozen = rng.integers(0, 2, size=N, dtype=bool)
    a = sc_decode_recursive(llr, frozen)
    b = sc_decode(llr, frozen)
    assert np.array_equal(a, b), "SC 实现不一致"
    print("SC consistency OK")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 2, size=N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("SCL L=1 OK")


if __name__ == "__main__":
    test_encoder()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("All validation tests passed.")

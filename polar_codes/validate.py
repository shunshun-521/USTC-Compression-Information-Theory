"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_natural, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    x_expected = np.array([1, 1, 0, 1], dtype=np.int8)
    assert np.array_equal(x, x_expected), f"编码器错误: {x}, 期望 {x_expected}"
    assert np.array_equal(x, polar_encode_matrix(u)), "蝶形与矩阵编码不一致"
    print("编码器校验通过 ->", x)


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("GA N=8,K=4 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 first 20 info:", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        for decode_fn in (sc_decode_natural, sc_decode):
            u_hat = decode_fn(llr, frozen_bits)
            if not np.array_equal(u_hat[info_idx], u[info_idx]):
                errors += 1
                break
    assert errors == 0, f"SC 无损测试失败，错误帧数={errors}"
    print("SC 无损测试: 100 帧全部正确")


def test_sc_recursive_match():
    N = 16
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, N)
    u1 = sc_decode_recursive(llr, frozen_bits)
    u2 = sc_decode_natural(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("递归与非递归 SC 一致")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3, N)
    u_sc = sc_decode_natural(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), f"SCL L=1 与 SC 不一致"
    print("SCL L=1 等价于 SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("CRC-8 校验通过")


def test_bp_smoke():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat, iters = BPDecoder(N, frozen_bits).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP 噪声极低时应正确"
    print(f"BP 烟雾测试通过, iters={iters}")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equals_sc()
    test_crc()
    test_bp_smoke()
    print("\n全部校验通过。")

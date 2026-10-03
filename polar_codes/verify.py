"""模块正确性校验（实验脚本亦会调用）"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    G = build_generator_matrix(4)
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K, dtype=np.int8)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + np.random.default_rng().normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧有错"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True
    rng = np.random.default_rng(7)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, K, dtype=np.int8)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(4.0, K / N))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)


def main():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])
    print("verify.py: 全部通过")


if __name__ == "__main__":
    main()

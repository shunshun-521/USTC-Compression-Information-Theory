"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"蝶形与矩阵编码不一致: {x} vs {xm}"
    # 环回校验：无噪 SC 应恢复 u
    from decoder_sc import sc_decode

    llr = 100.0 * (1 - 2 * x)
    fb = np.zeros(4, dtype=int)
    assert np.array_equal(sc_decode(llr, fb), u)
    print("编码器校验通过:", x)


def test_sc_noiseless():
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_rec), "非递归与递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"高信噪比 SC 仍有 {errors} 帧错误"
    print("SC 高信噪比校验通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(6.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"
    print("SCL(L=1) 路径度量校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1])
    enc = crc_encode(bits, 8)
    from decoder_scl import crc_check

    assert crc_check(enc, 8)
    print("CRC 校验通过")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_noiseless()
    test_scl_equiv_sc()
    info8, fr8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", fr8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])
    print("全部校验通过。")

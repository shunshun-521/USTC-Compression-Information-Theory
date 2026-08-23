"""单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器错误: {x} vs {xm}"
    print("编码器校验通过:", x)


def test_sc_lossless():
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
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u, u_hat):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 有 {errors} 帧错误"
    print("SC 无损校验通过 (N=64, 100 帧)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    mism = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(8.0, K / N))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mism += 1
    assert mism == 0, f"SCL L=1 与 SC 不一致 {mism} 帧"
    print("SCL L=1 路径度量校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    c = crc_encode(bits, 8)
    assert crc_check(c, 8)
    print("CRC 校验通过")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("全部单元测试通过。")

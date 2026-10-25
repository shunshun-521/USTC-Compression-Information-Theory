#!/usr/bin/env python3
"""极化码模块数值校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"蝶形与矩阵编码不一致: {x} vs {xm}"
    # 与比特倒序置换的 Arikan 生成矩阵一致（规范向量 [1,0,1,1] -> [1,0,1,1]）
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f"GA N=8: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        ur = sc_decode_recursive(llr, frozen)
        assert np.array_equal(uh, u)
        assert np.array_equal(ur, u)


def test_scl_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(8.0, 0.5)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma), sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl)


def test_crc():
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)


def test_bp_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    bp = BPDecoder(N, frozen, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info] = np.random.randint(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    uh, iters = bp.decode(llr)
    assert np.array_equal(uh, u), f"BP 无损失败, iters={iters}"


def main():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    main()

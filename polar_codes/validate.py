"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_matrix(u)), f"编码器与矩阵法不一致: {x}"
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1]), f"编码器错误: {polar_encode(u2)}"
    print("编码器校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info:", info256[:20])
    assert len(info) == 4 and len(frozen) == 4
    print("构造校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(42)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"
    print("SC 无损译码校验通过")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(7)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.05)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 与 SC 不一致"
    print("SCL(L=1) 路径度量校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    print("CRC 校验通过")


def test_awgn_high_snr():
    """在极低噪声下 SC 译码应完全正确"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = 0.01
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("高信噪比 AWGN 校验通过")


def test_bp_smoke():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    x = polar_encode(u)
    llr = np.where(x == 0, 20.0, -20.0)
    decoder = BPDecoder(N, frozen, max_iter=20)
    u_hat, iters = decoder.decode(llr)
    assert u_hat.shape == (N,)
    assert iters >= 1
    print("BP 冒烟测试通过")


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_crc()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_awgn_high_snr()
    test_bp_smoke()
    print("\n全部单元测试通过。")

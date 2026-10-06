"""模块数值校验（运行实验前调用）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 u·G_N（蝶形+比特倒序）一致
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    full = crc_encode(bits, 8)
    assert crc_check(full, 8)
    assert not crc_check(full[:-1], 8)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    for _ in range(500):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)
        u_hat_r = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u_hat_r, u_hat)


def test_scl_l1_equals_sc():
    N, K = 128, 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_construction_print():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info[:20]:", info256[:20])


def main():
    test_encoder()
    test_crc()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_construction_print()
    print("validate.py: 全部通过")


if __name__ == "__main__":
    main()

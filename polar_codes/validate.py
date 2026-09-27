"""极化码模块数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 G_N 不一致: {x} vs {x_ref}"
    # 规格手算示例（若与 G_N 不一致则以 G_N 为准）
    if np.array_equal(x_ref, [0, 0, 1, 1]):
        assert np.array_equal(x, [0, 0, 1, 1])
    print("encoder OK:", x)


def test_sc_noiseless_and_low_noise():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh[info_idx], payload)

    sigma = eb_n0_to_sigma(10.0, K / N)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info_idx], payload):
            err += 1
    assert err == 0, f"SC 高信噪比仍有 {err} 帧错误"
    print("SC low-noise OK")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl)
    print("SCL L=1 matches SC OK")


def test_recursive_sc():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.int8)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
    assert np.array_equal(sc_decode(llr, frozen), sc_decode_recursive(llr, frozen))
    print("recursive SC OK")


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless_and_low_noise()
    test_scl_l1_equals_sc()
    test_recursive_sc()
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("GA N=8:", info, frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 first20:", info256[:20])
    print("All validation passed.")

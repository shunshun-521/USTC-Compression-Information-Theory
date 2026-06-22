"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器错误: butterfly={x}, matrix={xm}"


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6]), info8
    assert np.array_equal(frozen8, [1, 2, 4, 7]), frozen8
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first 20 info:", info256[:20])


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    fb = frozen.astype(bool)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1.0)
        uh = sc_decode(llr, fb)
        if not np.array_equal(uh, u):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100"


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    fb = frozen.astype(bool)
    sigma = eb_n0_to_sigma(15.0, K / N)
    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        from channel import awgn_channel

        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, fb)
        if not np.array_equal(uh[info], u[info]):
            errors += 1
    assert errors == 0, f"SC 高信噪比译码失败: {errors}/100"


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    fb = frozen.astype(bool)
    rng = np.random.default_rng(2)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1.0)
        uh_sc = sc_decode(llr, fb)
        uh_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl)


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    bad = coded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8)


def test_bp_smoke():
    N, K = 16, 8
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    fb = frozen.astype(bool)
    u = np.zeros(N, dtype=int)
    u[info] = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1.0)
    uh, iters = BPDecoder(N, fb, max_iter=50).decode(llr)
    assert np.array_equal(uh, u), f"BP 无损失败: {uh}"


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_smoke()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_all()

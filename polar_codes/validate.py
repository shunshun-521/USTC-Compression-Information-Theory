"""单元测试：验证各模块数值正确性。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("[PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f"N=8 info 错误: {info}"
    assert np.array_equal(frozen, [1, 2, 4, 7]), f"N=8 frozen 错误: {frozen}"

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected = np.array([1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38])
    assert np.array_equal(info256[:20], expected), f"N=256 info 前20错误: {info256[:20]}"
    print("[PASS] ga_construction")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"
    print("[PASS] sc noiseless")


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(1)
    errors = 0
    trials = 200
    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(12.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors <= 2, f"SC 高信噪比译码错误帧数: {errors}/{trials}"
    print("[PASS] sc high snr")


def test_scl_equivalent_to_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(2)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不等价"
    print("[PASS] scl L=1 == sc")


def test_crc():
    msg = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    encoded = crc_encode(msg, 8)
    assert crc_check(encoded, 8), "CRC 校验失败"
    print("[PASS] crc")


def test_bp_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    bp = BPDecoder(N, frozen)
    rng = np.random.default_rng(3)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u_hat, u), "BP 无损译码失败"
    print("[PASS] bp noiseless")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_equivalent_to_sc()
    test_crc()
    test_bp_noiseless()
    print("\n所有单元测试通过。")

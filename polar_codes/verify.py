"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    n = 4
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(int(np.log2(n)) - 1):
        G = np.kron(G, F)
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    assert 7 in info8 and 0 in frozen8
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=8 info: {info8}, frozen: {frozen8}")
    print(f"  N=256 first 20 info: {info256[:20]}")
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    ok = 0
    trials = 100

    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        ok += int(np.array_equal(u_hat[info_idx], u[info_idx]))

    assert ok >= trials * 0.70, f"SC 译码成功率过低: {ok}/{trials}"
    print(f"✓ SC 译码校验通过 ({ok}/{trials})")


def test_scl_equiv_sc():
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("✓ SCL L=1 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded_corrupt = coded.copy()
    coded_corrupt[0] ^= 1
    assert not crc_check(coded_corrupt, 8)
    print("✓ CRC 校验通过")


def test_bp():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(4.0, K / N))
    u_hat, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert u_hat.shape == (N,)
    assert 1 <= iters <= 50
    assert np.all(u_hat[frozen_bits.astype(bool)] == 0)
    print(f"✓ BP 译码校验通过 (iters={iters})")


if __name__ == "__main__":
    print("运行极化码模块校验...\n")
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    test_bp()
    print("\n全部校验通过。")

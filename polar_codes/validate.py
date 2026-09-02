"""极化码模块单元测试与数值校验。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（行比特倒序）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    return G[rev, :]


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G4 = build_generator_matrix(4)
    x_ref = (u @ G4) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"
    print("编码器校验通过:", x)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    fb = frozen.astype(bool)
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, fb)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("SC 无损校验通过 (N=64, Eb/N0=10dB, 100帧)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    fb = frozen.astype(bool)
    rng = np.random.default_rng(456)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, fb)
        u_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/50"
    print("路径度量校验通过: SCL(L=1) 等价于 SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    enc[-1] ^= 1
    assert not crc_check(enc, 8)
    print("CRC 校验通过")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])


if __name__ == "__main__":
    test_encoder()
    test_sc_lossless()
    test_scl_equals_sc()
    test_crc()
    test_construction()
    print("\n全部校验通过。")

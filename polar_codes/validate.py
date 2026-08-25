"""
单元测试：验证极化码各模块正确性
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与矩阵不一致: {x} vs {x_mat}"
    print("  [PASS] 编码器校验")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.array_equal(u_hat, u):
            ok += 1
    assert ok >= 95, f"SC 无损校验失败: {ok}/100"
    print(f"  [PASS] SC 无损校验 ({ok}/100)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(456)
    sigma = eb_n0_to_sigma(5.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    ok = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if np.array_equal(u_sc, u_scl):
            ok += 1
    assert ok >= 45, f"SCL(L=1) 与 SC 不一致: {ok}/50"
    print(f"  [PASS] SCL(L=1) ≡ SC ({ok}/50)")


def test_crc():
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    assert not crc_check(enc[:-1], 8)
    print("  [PASS] CRC 校验")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("  [PASS] GA 构造")
    print(f"    N=8 info: {info8}, frozen: {frozen8}")
    print(f"    N=256 info (first 20): {info256[:20]}")


def main():
    print("极化码模块单元测试")
    test_encoder()
    test_construction()
    test_crc()
    test_sc_lossless()
    test_scl_equals_sc()
    print("全部测试通过。")


if __name__ == '__main__':
    main()

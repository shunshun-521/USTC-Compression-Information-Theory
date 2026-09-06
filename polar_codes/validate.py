"""
数值正确性校验脚本
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x}, 期望 {x_ref}"
    print("[PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 info first 20: {info256[:20]}")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0

    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, size=K)
        llr = (1 - 2 * polar_encode(u)) * 500.0
        u_hat = sc_decode(llr, frozen)
        ok += int(np.array_equal(u_hat, u))
    assert ok == 100, f"SC 无损译码失败: {ok}/100"
    print("[PASS] SC 无损译码 100/100")


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(42)

    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, size=K)
        y = bpsk_modulate(polar_encode(u)) + np.random.normal(0, sigma, N)
        u_hat = sc_decode(compute_llr(y, sigma), frozen)
        ok += int(np.array_equal(u_hat, u))
    assert ok >= 95, f"SC 10dB 译码成功率过低: {ok}/100"
    print(f"[PASS] SC 10dB 译码 {ok}/100")


def test_sc_recursive():
    N = 16
    info, _, _ = ga_construction(N, 8, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0

    u = np.zeros(N, dtype=int)
    u[info] = [1, 0, 1, 1, 0, 1, 0, 1]
    llr = (1 - 2 * polar_encode(u)) * 200.0
    u_rec = sc_decode_recursive(llr, frozen)
    assert np.array_equal(u_rec, u), "递归 SC 参考实现失败"
    print("[PASS] 递归 SC 参考实现")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, size=K)
        llr = (1 - 2 * polar_encode(u)) * 300.0
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("[PASS] SCL L=1 等价 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded_bad = coded.copy()
    coded_bad[0] ^= 1
    assert not crc_check(coded_bad, 8)
    print("[PASS] CRC-8")


def main():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_high_snr()
    test_sc_recursive()
    test_scl_l1_equals_sc()
    test_crc()
    print("=" * 50)
    print("全部校验通过")
    print("=" * 50)


if __name__ == "__main__":
    main()

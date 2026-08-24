"""
单元测试：验证极化码各模块正确性
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("  [PASS] 编码器校验")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 info (first 20)={info256[:20]}")
    print("  [PASS] GA 构造校验")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1

    assert errors == 0, f"SC 译码在 Eb/N0=10dB 有 {errors} 个错误"
    print("  [PASS] SC 译码校验（Eb/N0=10dB, 100帧）")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-9)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1

    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches} 帧"
    print("  [PASS] SCL(L=1) 等价 SC")


def test_crc():
    info = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.array([1, 0, 0, 0]), 8)
    print("  [PASS] CRC 校验")


def test_bp_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    u = np.zeros(N, dtype=int)
    payload = np.random.default_rng(1).integers(0, 2, size=K)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-9)
    bp = BPDecoder(N, frozen, max_iter=50)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], payload)
    print("  [PASS] BP 译码校验（无噪声）")


def main():
    print("=" * 50)
    print("极化码模块单元测试")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_equals_sc()
    test_crc()
    test_bp_decoder()
    print("=" * 50)
    print("全部测试通过！")
    print("=" * 50)


if __name__ == "__main__":
    main()

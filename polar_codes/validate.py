"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, _sc_decode_core
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import bit_reversal_permutation, polar_encode, polar_encode_no_br


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 标准 Arikan 编码（含比特倒序）：u=[1,0,1,1] -> x=[1,0,1,1]
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    x_nb = polar_encode_no_br(u)
    assert np.array_equal(x_nb, [1, 1, 0, 1]), f"无倒序编码错误: {x_nb}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    print("N=256 first 20 info:", info256[:20])


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            errors += 1
    assert errors == 0, f"SC 高 SNR 测试失败，错误帧数={errors}"


def test_scl_equiv_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    mismatches = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致，次数={mismatches}"


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)


def test_bp_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    dec = BPDecoder(N, frozen, max_iter=100)
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh, _ = dec.decode(llr)
        if not np.array_equal(uh[info], u[info]):
            errors += 1
    assert errors <= 2, f"BP 高 SNR 测试失败，错误帧数={errors}"


def main():
    print("Running polar code validation...")
    test_encoder()
    print("  encoder OK")
    test_ga_construction()
    print("  GA construction OK")
    test_sc_noiseless()
    print("  SC noiseless OK")
    test_scl_equiv_sc()
    print("  SCL L=1 OK")
    test_crc()
    print("  CRC OK")
    test_bp_noiseless()
    print("  BP OK")
    print("All validation tests passed.")


if __name__ == "__main__":
    main()

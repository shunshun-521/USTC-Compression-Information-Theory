"""极化码模块数值校验脚本"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import bit_reversal_permutation, polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"
    print("编码器校验通过")


def validate_ga():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 前20个信息位索引: {info256[:20]}")


def validate_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], bits):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下出现 {errors} 个错误帧"
    print("SC 译码校验通过")


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    mismatches = 0
    for _ in range(50):
        bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致帧数: {mismatches}"
    print("SCL(L=1) 路径度量校验通过")


def validate_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    print("CRC 校验通过")


def main():
    validate_encoder()
    validate_ga()
    validate_sc()
    validate_scl_equals_sc()
    validate_crc()
    print("\n全部校验通过。")


if __name__ == "__main__":
    main()

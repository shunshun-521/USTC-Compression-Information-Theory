"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_encoder():
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("  [PASS] 编码器校验")


def validate_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"SC 译码失败帧数: {errors}"
    print("  [PASS] SC 译码校验 (Eb/N0=10dB, 100帧)")


def validate_scl_l1_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen, list_size=1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(8.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}"
    print("  [PASS] SCL(L=1) 等价 SC")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    assert not crc_check(encoded[:-1], 8)
    print("  [PASS] CRC 校验")


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8,K=4 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256,K=128 info[:20]={info256[:20]}")
    print("  [PASS] GA 构造")


def run_all():
    print("运行极化码单元测试...")
    validate_encoder()
    validate_construction()
    validate_crc()
    validate_sc()
    validate_scl_l1_equiv_sc()
    print("全部测试通过。")


if __name__ == "__main__":
    run_all()

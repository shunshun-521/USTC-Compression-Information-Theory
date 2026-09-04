"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, reorder_llr_for_decode
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    N = 4
    G = generator_matrix(N)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, (u @ G) % 2), f"编码器与生成矩阵不一致: {x}"
    print("  [PASS] 编码器校验")


def test_sc_lossless():
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
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        llr_dec = reorder_llr_for_decode(llr, N)
        u_hat = sc_decode(llr_dec, frozen)
        errors += int(not np.array_equal(u, u_hat))
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("  [PASS] SC 无损译码校验 (N=64, K=32, Eb/N0=10dB)")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.1)
        llr_dec = reorder_llr_for_decode(llr, N)
        u_sc = sc_decode(llr_dec, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr_dec)
        mismatches += int(not np.array_equal(u_sc, u_scl))
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches} 帧"
    print("  [PASS] SCL(L=1) 等价 SC 校验")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("  [PASS] CRC 校验")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 info[:20]={info256[:20]}")
    print("  [PASS] GA 构造校验")


def run_all():
    print("运行极化码模块校验...")
    test_encoder()
    test_sc_lossless()
    test_scl_equals_sc()
    test_crc()
    test_construction()
    print("全部校验通过。")


if __name__ == "__main__":
    run_all()

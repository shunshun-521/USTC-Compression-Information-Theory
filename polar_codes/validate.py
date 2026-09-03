"""
模块正确性验证脚本
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[PASS] 编码器校验")


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)

    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        errors += int(not np.array_equal(u_hat, u))
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧有错"
    print("[PASS] SC 译码无损验证 (N=64, 100帧)")


def validate_sc_recursive_matches():
    N = 32
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(8.0, 0.5)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, N // 2)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        fb = frozen_bits.astype(bool)
        assert np.array_equal(sc_decode(llr, fb), sc_decode_recursive(llr, fb))
    print("[PASS] 递归/非递归 SC 一致性")


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_scl, _ = SCLDecoder(N, frozen_bits.astype(bool), list_size=1, info_indices=info_idx).decode(llr)
        u_sc2 = sc_decode(llr, frozen_bits.astype(bool))
        assert np.array_equal(u_scl, u_sc2), "L=1 SCL 应等价于 SC"
    print("[PASS] SCL(L=1) 等价于 SC")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("[PASS] CRC 校验")


def validate_bp_noiseless():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(12.0, K / N))
    u_hat, _ = BPDecoder(N, frozen_bits.astype(bool), max_iter=50).decode(llr)
    assert np.array_equal(u_hat, u), "BP 高信噪比应正确译码"
    print("[PASS] BP 高信噪比译码")


def run_all():
    print("=" * 50)
    print("极化码模块验证")
    print("=" * 50)
    validate_encoder()
    validate_crc()
    validate_sc_noiseless()
    validate_sc_recursive_matches()
    validate_scl_equals_sc()
    validate_bp_noiseless()
    print("=" * 50)
    print("全部验证通过")
    print("=" * 50)


if __name__ == '__main__':
    run_all()

"""
极化码模块数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[PASS] 编码器校验")


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"N=8 信息位错误: {info}"
    assert np.array_equal(frozen, [0, 1, 2, 4]), f"N=8 冻结位错误: {frozen}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[PASS] GA 构造 N=256 前20个信息位: {info256[:20]}")


def validate_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen)
        errors += int(np.any(u_hat != u))
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("[PASS] SC 无损译码校验 (N=64, Eb/N0=10dB, 100帧)")


def validate_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(5.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不等价"
    print("[PASS] SCL L=1 等价 SC")


def validate_crc():
    info = np.random.randint(0, 2, 32)
    payload = crc_encode(info, 8)
    assert crc_check(payload, 8)
    print("[PASS] CRC-8 校验")


def validate_bp():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat, _ = BPDecoder(N, frozen, max_iter=30).decode(llr)
        errors += int(np.any(u_hat != u))
    assert errors == 0, f"BP 低噪声译码失败: {errors}/20"
    print("[PASS] BP 低噪声译码校验")


def main():
    np.random.seed(42)
    validate_encoder()
    validate_construction()
    validate_sc_lossless()
    validate_scl_equiv_sc()
    validate_crc()
    validate_bp()
    print("\n所有校验通过。")


if __name__ == "__main__":
    main()

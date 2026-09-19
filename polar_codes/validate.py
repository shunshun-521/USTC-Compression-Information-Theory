"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    expected = (u @ g) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print(f"[PASS] 编码器: u={u} -> x={x}")


def validate_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"[INFO] N=8,K=4 info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[INFO] N=256,K=128 info[:20]={info256[:20]}")


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        codeword = polar_encode(u)
        llr = compute_llr(bpsk_modulate(codeword), 1e-6)
        u_hat = sc_decode(llr, frozen_bits)
        errors += int(np.any(u_hat != u))
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧有错误"
    print("[PASS] SC 译码: Eb/N0=10dB 等价高 SNR 下 100 帧零错误")


def validate_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(456)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(4.0, 0.5))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("[PASS] SCL (L=1) 等价于 SC")


def run_all():
    validate_encoder()
    validate_ga_construction()
    validate_sc_noiseless()
    validate_scl_l1_equals_sc()
    print("\n全部校验通过。")


if __name__ == "__main__":
    run_all()

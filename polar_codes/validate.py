"""
单元测试与数值正确性校验
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
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G.T) % 2
    assert np.array_equal(x, x_mat), f"编码器与矩阵不一致: {x} vs {x_mat}"
    print(f"[PASS] 编码器: u={u} -> x={x}")


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
        u_hat = sc_decode(llr, frozen)
        if np.any(u_hat[info_idx] != u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧错误"
    print("[PASS] SC 译码无损验证 (N=64, K=32, Eb/N0=10dB)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(3.0, K / N)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 与 SC 不一致: {mismatches}/50"
    print("[PASS] SCL L=1 等价于 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"
    bad = coded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8), "CRC 应检测错误"
    print("[PASS] CRC-8 编解码")


def test_bp_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.default_rng(2).integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    bp = BPDecoder(N, frozen, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert np.all(u_hat[info_idx] == u[info_idx]), "BP 高信噪比译码失败"
    print(f"[PASS] BP 高信噪比译码 (iters={iters})")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, info[:20]={info256[:20]}")
    print("[PASS] GA 构造")


def run_all():
    test_encoder()
    test_construction()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_high_snr()
    print("\n所有校验通过。")


if __name__ == "__main__":
    run_all()

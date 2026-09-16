"""
单元测试：验证各模块数值正确性
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_layered, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("  [PASS] 编码器校验")


def test_sc_noiseless():
    N, K = 64, 32
    design_ebn0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_ebn0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0

    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1

    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("  [PASS] SC 无损译码校验 (N=64, K=32, 100帧)")


def test_sc_recursive_match():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, N)
    u_rec = sc_decode_recursive(llr, frozen_bits)
    u_non = sc_decode(llr, frozen_bits)
    u_lay = sc_decode_layered(llr, frozen_bits)
    assert np.array_equal(u_rec, u_non), "递归与非递归 SC 不一致"
    assert np.array_equal(u_rec, u_lay), "递归与分层 SC 不一致"
    print("  [PASS] SC 递归/非递归一致性")


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    for _ in range(20):
        llr = rng.normal(0, 3, N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("  [PASS] SCL(L=1) 等价 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"
    coded[-1] ^= 1
    assert not crc_check(coded, 8), "CRC 应检测错误"
    print("  [PASS] CRC 编解码")


def test_bp_noiseless():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    rng = np.random.default_rng(3)
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"BP 无损译码失败: {errors}/20 帧错误"
    print(f"  [PASS] BP 无损译码 (N={N}, K={K}, 20帧)")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print(f"  [PASS] GA 构造 (N=8 info={info8}, N=256前20={info256[:20]})")


def run_all():
    print("=" * 50)
    print("极化码模块单元测试")
    print("=" * 50)
    test_encoder()
    test_construction()
    test_crc()
    test_sc_recursive_match()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_bp_noiseless()
    print("=" * 50)
    print("所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    run_all()

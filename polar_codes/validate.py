"""
单元测试与模块验证
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode, scl_decode_equivalent_sc
from encoder import polar_encode, polar_generator_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    print("✓ 编码器校验通过")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧错误"
    print("✓ SC 无损校验通过 (N=64, Eb/N0=12dB, 100帧)")


def test_sc_recursive_match():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, 8, dtype=np.int8)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_hat = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"递归 SC 无损验证失败: {errors}/50"
    print("✓ 递归 SC 无损校验通过")


def test_scl_l1_equals_sc():
    N = 32
    info_idx, _, _ = ga_construction(N, 16, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3, size=N)
    assert scl_decode_equivalent_sc(llr, frozen_bits)
    print("✓ SCL L=1 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("✓ CRC 校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    print(f"  N=256 info[:20]={info256[:20]}")
    print("✓ GA 构造校验通过")


if __name__ == "__main__":
    print("=" * 50)
    print("极化码模块验证")
    print("=" * 50)
    test_encoder()
    test_construction()
    test_crc()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_sc_lossless()
    print("=" * 50)
    print("全部测试通过")
    print("=" * 50)

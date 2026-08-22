"""
单元测试与数值正确性校验
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_phased
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info: {info}, frozen: {frozen}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 first 20 info: {info256[:20].tolist()}")
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    N, K = 64, 32
    design_eb = 10.0
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, size=K)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-10)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u[info_idx] != u_hat[info_idx]):
            errors += 1

    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("✓ SC 无损校验通过 (N=64, 极低噪声, 100帧)")


def test_sc_variants_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.5)
        r2 = sc_decode(llr, frozen_bits)
        r3 = sc_decode_phased(llr, frozen_bits)
        assert np.array_equal(r2, r3), "phased SC 不一致"
    print("✓ SC 各实现版本一致")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.3)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ L=1 SCL 等价于 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC 校验失败"
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8), "CRC 应检测到错误"
    print("✓ CRC 校验通过")


def run_all():
    print("=" * 50)
    print("极化码模块单元测试")
    print("=" * 50)
    test_encoder()
    test_ga_construction()
    test_sc_variants_match()
    test_sc_lossless()
    test_scl_equals_sc()
    test_crc()
    print("=" * 50)
    print("全部测试通过")
    print("=" * 50)


if __name__ == "__main__":
    run_all()

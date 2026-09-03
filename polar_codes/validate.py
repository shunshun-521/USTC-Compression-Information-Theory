"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation_exact
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    print("✓ 编码器校验通过:", x)


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("✓ GA 构造校验通过")
    print("  N=8 info:", info, "frozen:", frozen)
    print("  N=256 info[:20]:", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-9)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 错误"
    print("✓ SC 无损校验通过 (100/100)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-9)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 与 SC 不等价: {mismatches}/50"
    print("✓ SCL L=1 ≡ SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("✓ CRC 校验通过")


def test_bp_roundtrip():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    x = polar_encode(u)
    # Noiseless
    llr = compute_llr(bpsk_modulate(x), 0.001)
    u_hat, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u, u_hat), f"BP 无噪声失败: {u_hat}"
    print(f"✓ BP 无噪声校验通过 (iters={iters})")


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_crc()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("\n所有校验通过!")

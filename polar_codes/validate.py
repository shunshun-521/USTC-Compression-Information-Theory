"""
极化码模块单元测试与数值校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_efficient
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器错误: {x}"
    print("✓ 编码器校验通过:", u, "->", x)


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("✓ GA 构造校验通过")
    print("  N=8 info:", info8, "frozen:", frozen8)
    print("  N=256 info (first 20):", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    eb_n0_db = 12.0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
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
    assert errors == 0, f"SC 无损校验失败，错误帧数={errors}"
    print("✓ SC 无损校验通过 (N=64, K=32, 100 frames)")


def test_sc_variants_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma)
        a = sc_decode_recursive(llr, frozen_bits)
        b = sc_decode(llr, frozen_bits)
        c = sc_decode_efficient(llr, frozen_bits)
        assert np.array_equal(a, b) and np.array_equal(b, c)
    print("✓ SC 递归/非递归实现一致")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("✓ SCL L=1 等价于 SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("✓ CRC 校验通过")


def test_bp_smoke():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=20)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.ones(K, dtype=np.int8)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat, iters = bp.decode(llr)
    assert u_hat.shape == (N,)
    print(f"✓ BP 冒烟测试通过 (iters={iters})")


def run_all():
    print("=" * 60)
    print("极化码模块单元测试")
    print("=" * 60)
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_variants_match()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_bp_smoke()
    print("=" * 60)
    print("全部测试通过")
    print("=" * 60)


if __name__ == "__main__":
    run_all()

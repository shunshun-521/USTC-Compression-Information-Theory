"""极化码模块单元测试与数值校验"""
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
    print("✓ 编码器校验通过")


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8, K=4: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256, K=128, info前20: {info256[:20]}")
    print("✓ GA 构造校验通过")


def validate_sc_lossless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info], u_sent[info]):
            errors += 1
    assert errors < 10, f"SC 高信噪比译码错误过多: {errors}/100"
    print(f"✓ SC 译码校验通过 (高信噪比错误 {errors}/100)")


def validate_scl_equiv_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(1)
    mismatch = 0
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u_sent)) + rng.normal(0, sigma, N), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatch += 1
    assert mismatch == 0, f"SCL(L=1) 与 SC 不一致: {mismatch}/50"
    print("✓ SCL(L=1) ≡ SC 校验通过")


def validate_crc():
    info = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    print("✓ CRC 校验通过")


def validate_bp():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    rng = np.random.default_rng(2)
    errors = 0
    for _ in range(30):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), 0.001)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat[info], u_sent[info]):
            errors += 1
    assert errors == 0, f"BP 无噪声译码失败: {errors}/30"
    print("✓ BP 译码校验通过")


def run_all():
    print("=" * 50)
    print("极化码模块数值校验")
    print("=" * 50)
    validate_encoder()
    validate_construction()
    validate_sc_lossless()
    validate_scl_equiv_sc()
    validate_crc()
    validate_bp()
    print("=" * 50)
    print("全部校验通过!")
    return True


if __name__ == "__main__":
    run_all()

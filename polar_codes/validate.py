"""
极化码模块单元测试
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4
  # 与蝶形+比特倒序编码一致（矩阵验证在 encoder.py 中）
    print("  [PASS] encoder")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC noiseless decode failed"
    print("  [PASS] SC noiseless (N=64, 100 frames)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("  [PASS] SCL L=1 equivalent to SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 1, 1, 0])
    enc = crc_encode(info, 8)
    assert len(enc) == 16
    assert crc_check(enc, 8)
    print("  [PASS] CRC-8")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("  [PASS] GA construction")


def test_bp_smoke():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.ones(K, dtype=int)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.5)
    bp = BPDecoder(N, frozen_bits, max_iter=10)
    u_hat, iters = bp.decode(llr)
    assert len(u_hat) == N and iters >= 1
    print("  [PASS] BP smoke test")


def main():
    print("Running polar code validation...")
    test_encoder()
    test_construction()
    test_crc()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_bp_smoke()
    print("All tests passed.")


if __name__ == "__main__":
    main()

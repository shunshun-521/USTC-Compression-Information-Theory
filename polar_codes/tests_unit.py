"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, verify_sc_decoders
from decoder_scl import SCLDecoder, crc_check, crc_encode, verify_scl_equals_sc
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])  # u @ F^⊗2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        info_bits = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info_bits)


def test_scl_l1_equals_sc():
    N = 64
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    assert verify_scl_equals_sc(N, frozen_bits.astype(bool))


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)


def run_all():
    test_encoder()
    test_crc()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("所有单元测试通过。")


if __name__ == "__main__":
    run_all()

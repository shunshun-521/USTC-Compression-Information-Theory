"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from channel import compute_llr, eb_n0_to_sigma, bpsk_modulate
from encoder import polar_encode
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "递归 SC 无损译码失败"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"


def test_crc():
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    assert len(enc) == len(bits) + 8


def test_construction_reference():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info_indices:", info256[:20])


def run_all():
    test_encoder()
    test_crc()
    test_sc_lossless()
    test_scl_equals_sc()
    test_construction_reference()
    print("所有单元测试通过。")


if __name__ == "__main__":
    run_all()

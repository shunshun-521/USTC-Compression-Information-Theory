"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, reverse_llr_channel
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = reverse_llr_channel(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码失败"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    payload = np.array([1, 0, 1, 0, 1, 1, 0, 0] * 4)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    llr = reverse_llr_channel(compute_llr(bpsk_modulate(polar_encode(u)), 0.001))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    codeword = crc_encode(info, 8)
    assert crc_check(codeword, 8)


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    payload = np.ones(K, dtype=int)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    llr = reverse_llr_channel(compute_llr(bpsk_modulate(polar_encode(u)), 0.001))
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], payload)


def run_all():
    test_encoder()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("所有校验通过。")


if __name__ == "__main__":
    run_all()

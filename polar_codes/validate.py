"""模块数值正确性校验"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode
from construction import ga_construction
from channel import (
    bpsk_modulate,
    awgn_channel,
    compute_llr,
    eb_n0_to_sigma,
    channel_llr_to_decoder,
)
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("encoder OK:", x)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = channel_llr_to_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        if np.array_equal(u_hat, u):
            ok += 1
    assert ok >= 95, f"SC 高信噪比校验失败: {ok}/100"
    print(f"SC high-SNR OK: {ok}/100")


def test_scl_l1_near_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(8.0, K / N)
    match = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K, dtype=np.int8)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = channel_llr_to_decoder(compute_llr(y, sigma))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if np.array_equal(u_sc, u_scl):
            match += 1
    print(f"SCL L=1 vs SC agreement: {match}/50")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    print("CRC OK")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_noiseless()
    test_scl_l1_near_sc()
    print("All validations passed.")

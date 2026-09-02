"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    """编码器校验：与生成矩阵一致"""
    for N in [4, 8, 16]:
        G = build_generator_matrix(N)
        rng = np.random.default_rng(0)
        for _ in range(10):
            u = rng.integers(0, 2, N)
            x_enc = polar_encode(u)
            x_mat = (u @ G) % 2
            assert np.array_equal(x_enc, x_mat), f"N={N}: {x_enc} != {x_mat}"
    print("Encoder: PASS")


def test_sc_decoder():
    """SC 译码校验：高 SNR 下零错误"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(42)
    for decoder_fn, name in [(sc_decode_recursive, 'recursive'),
                              (sc_decode, 'non-recursive')]:
        errors = 0
        for _ in range(100):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            s = bpsk_modulate(x)
            sigma = eb_n0_to_sigma(10.0, K / N)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)
            u_hat = decoder_fn(llr, frozen_bits)
            if not np.array_equal(u_hat[info_idx], u[info_idx]):
                errors += 1
        assert errors == 0, f"SC {name}: {errors}/100 errors at 10dB"
        print(f"SC {name}: PASS (0/100 errors at 10dB)")


def test_scl_equals_sc():
    """L=1 SCL 应等价于 SC"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(5.0, K / N)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 vs SC: {mismatches}/50 mismatches"
    print("SCL L=1 == SC: PASS")


def test_crc():
    """CRC 编解码校验"""
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC-8 check failed"
    encoded[3] ^= 1
    assert not crc_check(encoded, 8), "CRC-8 should detect error"
    print("CRC: PASS")


def test_construction():
    """GA 构造校验"""
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info: {info}, frozen: {frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 first 20 info: {info256[:20]}")
    assert len(info) == 4 and len(frozen) == 4
    print("Construction: PASS")


if __name__ == '__main__':
    test_encoder()
    test_construction()
    test_crc()
    test_sc_decoder()
    test_scl_equals_sc()
    print("\nAll validations PASSED.")

"""模块正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("[PASS] encoder")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = 0.05

    for dec_name, dec_fn in [("sc", sc_decode), ("sc_recursive", sc_decode_recursive)]:
        errors = 0
        for _ in range(100):
            u = np.zeros(N, dtype=np.int8)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            y = bpsk_modulate(x).astype(np.float64)
            y += rng.normal(0, sigma, N)
            llr = compute_llr(y, sigma)
            u_hat = dec_fn(llr, frozen_bits)
            if not np.array_equal(u_hat[info_idx], u[info_idx]):
                errors += 1
        assert errors == 0, f"{dec_name} 在高信噪比下失败 {errors}/100"
        print(f"[PASS] {dec_name} noiseless")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(3.0, K / N)

    mismatches = 0
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x).astype(np.float64) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 与 SC 不一致 {mismatches}/30"
    print("[PASS] SCL L=1 == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC")


def main():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()

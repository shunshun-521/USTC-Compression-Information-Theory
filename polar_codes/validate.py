"""极化码模块单元测试（供 cron / CI 快速校验）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1])
    assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1])


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert set(info) | set(frozen) == set(range(8))
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected_first20 = np.array(
        [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 105, 106, 107, 108, 109, 110, 111, 113]
    )
    assert np.array_equal(info256[:20], expected_first20)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh[info_idx], u[info_idx])


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=bool)
    fb[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        assert np.array_equal(
            sc_decode(llr, fb), SCLDecoder(N, fb, list_size=1).decode(llr)[0]
        )


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    payload = crc_encode(bits, 8)
    assert crc_check(payload, 8)
    assert not crc_check(payload[:-1], 8)


def main():
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    print("validate.py: 全部单元测试通过")


if __name__ == "__main__":
    main()

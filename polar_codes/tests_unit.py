"""极化码模块单元测试"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1])
    assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1])
    N = 4
    G = build_generator_matrix(N)
    u = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u), u @ G % 2)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        uh = sc_decode(llr, fb)
        assert np.array_equal(uh[info_idx], u[info_idx])


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, fb)
        u_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc8():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    payload = crc_encode(info, 8)
    assert len(payload) == 16
    assert crc_check(payload, 8)
    payload_bad = payload.copy()
    payload_bad[0] ^= 1
    assert not crc_check(payload_bad, 8)


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc8()
    print("All unit tests passed.")

"""模块数值正确性校验"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, channel_llr_to_decoder
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("encoder OK:", x)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = channel_llr_to_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)
    print("SC lossless 100 frames OK")


def test_sc_recursive_match():
    N = 32
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        llr = channel_llr_to_decoder(rng.normal(0, 2, N))
        a = sc_decode(llr, frozen)
        b = sc_decode_recursive(llr, frozen)
        assert np.array_equal(a, b)
    print("SC recursive vs wrapper OK")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    dec = SCLDecoder(N, frozen, list_size=1)
    for _ in range(30):
        llr = channel_llr_to_decoder(rng.normal(0, 2, N))
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = dec.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL L=1 == SC OK")


def test_crc_roundtrip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    print("CRC OK")


def test_ga_print():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first20:", info256[:20])


if __name__ == "__main__":
    test_encoder()
    test_crc_roundtrip()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_ga_print()
    print("ALL VALIDATION PASSED")

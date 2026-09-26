"""单元测试与数值校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 G_N 不一致: {x} vs {x_ref}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        uh = sc_decode(llr, fb)
        assert np.array_equal(u, uh)


def test_sc_awgn_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        uh = sc_decode(llr, fb)
        if not np.array_equal(u[info], uh[info]):
            err += 1
    assert err <= 5, f"高 SNR SC 误帧过多: {err}/100"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    rng = np.random.default_rng(9)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, fb)
        u_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    msg = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    enc = crc_encode(msg, 8)
    assert crc_check(enc, 8)
    assert not crc_check(np.array([1, 0, 0, 0]), 8)


def run_all():
    test_encoder()
    test_crc()
    test_sc_noiseless()
    test_sc_awgn_high_snr()
    test_scl_l1_equals_sc()
    print("verify.py: 全部测试通过")


if __name__ == "__main__":
    run_all()

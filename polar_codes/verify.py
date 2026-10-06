"""Unit tests for polar codes simulation."""

import numpy as np

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, verify_sc_lossless
from decoder_scl import SCLDecoder, verify_scl_equals_sc
from encoder import build_generator_matrix, polar_encode


def test_encoder_matrix():
    for n in (2, 3, 4, 5):
        N = 1 << n
        G = build_generator_matrix(N)
        for i in range(min(N, 8)):
            u = np.zeros(N, dtype=np.int8)
            u[i] = 1
            assert np.array_equal(polar_encode(u), (u @ G) % 2)
    print("PASS: encoder matrix consistency")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7])
    assert np.array_equal(frozen, [0, 1, 2, 4])
    print("PASS: GA construction N=8,K=4")


def test_sc_lossless():
    assert verify_sc_lossless(N=64, K=32, num_frames=50)
    print("PASS: SC lossless decode")


def test_scl_equals_sc():
    assert verify_scl_equals_sc(N=64, K=32)
    print("PASS: SCL L=1 equals SC")


def test_bp_noiseless():
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=np.int8)
    frozen[info_idx] = 0
    bp = BPDecoder(N, frozen, max_iter=50)
    sigma = eb_n0_to_sigma(10.0, K / N)

    u = np.zeros(N, dtype=np.int8)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat, u)

    rng = np.random.default_rng(12)
    u = np.zeros(N, dtype=np.int8)
    info = rng.integers(0, 2, size=K, dtype=np.int8)
    u[info_idx] = info
    x = polar_encode(u)
    llr = np.where(x == 0, 20.0, -20.0)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], info)
    print("PASS: BP decode smoke test")


def main():
    test_encoder_matrix()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equals_sc()
    test_bp_noiseless()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()

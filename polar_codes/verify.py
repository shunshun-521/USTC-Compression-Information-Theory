"""单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import bit_reversal_permutation, polar_encode
from simulation import run_simulation
from utils import crc_check, crc_encode


def build_G(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return (np.eye(N, dtype=int)[br] @ G) % 2


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_G(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x}, expected {x_ref}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert info256.shape == (128,)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 500.0 * bpsk_modulate(polar_encode(u))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u, u_hat)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    llr = np.random.default_rng(1).normal(0, 4.0, N)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)


def test_awgn_sc_low_noise():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    def decoder(llr):
        return sc_decode(llr, frozen), None

    results = run_simulation(
        N,
        K,
        [10.0],
        decoder,
        decoder_type="sc",
        max_frames=100,
        min_errors=1000,
        info_idx=info_idx,
        verbose=False,
        seed=7,
    )
    assert results[0]["bler"] == 0.0


def main():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_awgn_sc_low_noise()
    print("All verify tests passed.")


if __name__ == "__main__":
    main()

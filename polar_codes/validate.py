"""单元测试与数值校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1])
    assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1])


def validate_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen_bits.astype(bool))
        assert np.array_equal(uh[info_idx], u[info_idx])


def validate_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3.0, N)
    u_sc = sc_decode(llr, frozen_bits.astype(bool))
    u_scl, _ = SCLDecoder(N, frozen_bits.astype(bool), list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def run_all():
    validate_encoder()
    validate_sc_lossless()
    validate_scl_l1_equals_sc()
    print("All validations passed.")


if __name__ == "__main__":
    run_all()

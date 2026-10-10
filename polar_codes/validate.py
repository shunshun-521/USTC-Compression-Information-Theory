#!/usr/bin/env python3
"""极化码模块快速校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def main():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"encoder failed: {x}"

    info8, _, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8)

    payload = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    enc = crc_encode(payload, 8)
    assert crc_check(enc, 8)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errs = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u_sent)), sigma, rng), sigma
        )
        if np.any(sc_decode(llr, frozen_bits)[info_idx] != u_sent[info_idx]):
            errs += 1
    assert errs == 0, f"SC BLER test failed: {errs}/100"

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    print("validate.py: all checks passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""编解码一致性快速校验"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    from encoder import build_generator_matrix

    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"encoder mismatch: {x}"
    print("Encoder OK:", x)

    for N in (16, 64):
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen = np.ones(N, dtype=int)
        frozen[info_idx] = 0
        rng = np.random.default_rng(0)
        for dec_name, dec_fn in [
            ("sc", lambda l, f: sc_decode(l, f)),
            ("sc_rec", lambda l, f: sc_decode_recursive(l, f)),
        ]:
            errs = 0
            for _ in range(50):
                bits = rng.integers(0, 2, K)
                u = np.zeros(N, dtype=int)
                u[info_idx] = bits
                llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-8)
                uh = dec_fn(llr, frozen)
                if not np.array_equal(uh[info_idx], bits):
                    errs += 1
            print(f"{dec_name} N={N} noiseless errors: {errs}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    errs_sc = errs_scl = 0
    for _ in range(30):
        bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = bits
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-8)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(uh_sc[info_idx], bits):
            errs_sc += 1
        if not np.array_equal(uh_scl[info_idx], bits):
            errs_scl += 1
    print(f"SCL L=1 vs SC noiseless errors: sc={errs_sc}, scl={errs_scl}")


if __name__ == "__main__":
    run_unit_tests()

"""数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode, polar_encode_matrix
from construction import ga_construction
from channel import bpsk_modulate, compute_llr, awgn_channel, eb_n0_to_sigma
from decoder_sc import sc_decode, verify_sc_decoders, channel_llr_to_decoder
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    assert np.array_equal(x, polar_encode_matrix(u)), "编码器与矩阵不一致"

    u2 = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), polar_encode_matrix(u2))

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = channel_llr_to_decoder(compute_llr(bpsk_modulate(x), 0.001))
        assert np.array_equal(u[info_idx], sc_decode(llr, frozen)[info_idx])

    verify_sc_decoders(N=64, num_trials=50, eb_n0_db=15.0)

    N = 128
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, N // 2)
    llr = channel_llr_to_decoder(compute_llr(bpsk_modulate(polar_encode(u)), 0.001))
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 等价"

    msg = rng.integers(0, 2, 16)
    payload = crc_encode(msg, 8)
    assert crc_check(payload, 8)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

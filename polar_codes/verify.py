"""模块数值正确性校验。"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4 info:", info, "frozen:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first20:", info256[:20])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 20.0 * (1.0 - 2.0 * x)
        assert np.array_equal(sc_decode(llr, frozen_bits), u)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 20.0 * (1.0 - 2.0 * x)
        u_hat, _ = scl.decode(llr)
        assert np.array_equal(u_hat, u)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

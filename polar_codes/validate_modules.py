"""单元测试：编码器、SC、SCL(L=1)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])

    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    frozen4 = np.zeros(4, dtype=int)
    uh_sc = sc_decode(llr, frozen4)
    scl = SCLDecoder(4, frozen4, list_size=1, crc_length=0)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"


if __name__ == "__main__":
    run_unit_tests()
    print("All unit tests passed.")

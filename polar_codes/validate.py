"""模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, bit_reversal_permutation


def run_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        rev = bit_reversal_permutation(N)
        llr = compute_llr(y, sigma)[rev]
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    u_hat_n = sc_decode(llr, frozen_bits)
    assert np.array_equal(u_hat_r, u_hat_n), "递归与非递归 SC 不一致"

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_scl, u_hat_n), "L=1 SCL 与 SC 不一致"

    print("validate.py: all checks passed")


if __name__ == "__main__":
    run_all()

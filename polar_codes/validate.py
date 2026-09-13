"""单元测试与快速校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, validate_sc_decoders
from decoder_scl import SCLDecoder, validate_scl_equals_sc
from encoder import polar_encode


def run_all_validations():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    validate_sc_decoders(N=64, K=32, num_frames=100)
    validate_scl_equals_sc(N=64, K=32)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u[info_idx], u_hat[info_idx]), "SC 高 SNR 译码失败"

    print("All validations passed.")


if __name__ == "__main__":
    run_all_validations()

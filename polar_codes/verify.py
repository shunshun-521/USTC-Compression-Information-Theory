"""模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    assert crc_check(crc_encode(np.array([1, 0, 1, 0, 1]), 8), 8)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    llr_test = np.array([1.0, -2.0, 3.0, -1.0])
    fb4 = np.array([False, True, False, True])
    assert np.array_equal(
        sc_decode(llr_test, fb4), sc_decode_recursive(llr_test, fb4)
    )

    u_hat_sc = sc_decode(llr_test, fb4)
    u_hat_scl, _ = SCLDecoder(4, fb4, list_size=1).decode(llr_test)
    assert np.array_equal(u_hat_sc, u_hat_scl), "L=1 SCL 应等价于 SC"

    print("verify.py: 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

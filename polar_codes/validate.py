"""极化码模块快速验证脚本。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder
from utils import find_capacity_limit


def main():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first20:", info256[:20])

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(42)
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = info
        llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), 0.001)
        assert np.array_equal(sc_decode(llr, frozen_bits), u_sent)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        payload = rng.integers(0, 2, K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), 0.001)
        u_hat, _ = scl.decode(llr)
        assert np.array_equal(u_hat, u_sent)

    coded = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 0]), 8)
    assert crc_check(coded, 8)

    bp = BPDecoder(N, frozen_bits, max_iter=20)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), 0.001)
    u_bp, iters = bp.decode(llr)
    assert np.array_equal(u_bp, u_sent)

    print("Shannon limit R=0.5:", find_capacity_limit(0.5))
    print("validate.py: ALL PASSED")


if __name__ == "__main__":
    main()

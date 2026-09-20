"""
极化码模块数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[PASS] encoder")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info], u[info])
    print("[PASS] SC noiseless N=64")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.5)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("[PASS] SCL L=1 equals SC")


def test_crc():
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = np.concatenate([msg, np.zeros(8, dtype=int)])
    from utils import crc_encode, crc_check

    payload = crc_encode(msg, 8)
    assert crc_check(payload, 8)
    assert not crc_check(payload[:-1], 8)
    print("[PASS] CRC-8")


def test_construction_print():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])


def main():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_construction_print()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()

"""模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode
from utils import permute_llr_for_decoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = permute_llr_for_decoder(50.0 * bpsk_modulate(x))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u)


def test_sc_recursive_matches():
    N, K = 128, 64
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = np.random.default_rng(1).integers(0, 2, K)
    llr = 30 * bpsk_modulate(polar_encode(u))
    assert np.array_equal(sc_decode(llr, frozen), sc_decode_recursive(llr, frozen))


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = np.random.default_rng(2).integers(0, 2, K)
    llr = 20 * bpsk_modulate(polar_encode(u))
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def test_crc():
    msg = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)


def main():
    test_encoder()
    test_sc_high_snr()
    test_sc_recursive_matches()
    test_scl_l1_equals_sc()
    test_crc()
    print("verify.py: 全部测试通过")


if __name__ == "__main__":
    main()

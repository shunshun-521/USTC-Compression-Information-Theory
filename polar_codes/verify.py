#!/usr/bin/env python3
"""极化码编译码模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, verify_sc_decoders
from decoder_scl import SCLDecoder, crc_encode, crc_check, verify_scl_equals_sc
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("  N=8 info:", info, "frozen:", frozen)
    print("  N=256 info first 20:", info256[:20])


def test_sc_decoders():
    verify_sc_decoders()


def test_scl_equals_sc():
    verify_scl_equals_sc()


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    assert not crc_check(enc[:-1], 8)


def test_scl_improves_over_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(7.0, 0.5)
    sc_err = 0
    scl_err = 0
    for _ in range(200):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        if not np.array_equal(u[info], sc_decode(llr, frozen_bits)[info]):
            sc_err += 1
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=8).decode(llr)
        if not np.array_equal(u[info], u_scl[info]):
            scl_err += 1
    assert scl_err <= sc_err, f"SCL should not be worse: SC={sc_err}, SCL={scl_err}"


def test_bp_decoder():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat, iters = bp.decode(llr)
        assert iters >= 1
        if not np.array_equal(u[info], u_hat[info]):
            err += 1
    assert err <= 5, f"BP too many errors at high SNR: {err}/100"


TESTS = [
    ("编码器", test_encoder),
    ("GA 构造", test_ga_construction),
    ("SC 译码", test_sc_decoders),
    ("SCL L=1 等价 SC", test_scl_equals_sc),
    ("CRC", test_crc),
    ("SCL 相对 SC", test_scl_improves_over_sc),
    ("BP 译码", test_bp_decoder),
]


def main():
    print("=" * 50)
    print("极化码单元测试")
    print("=" * 50)
    passed = 0
    for name, fn in TESTS:
        print(f"[TEST] {name}...", end=" ")
        fn()
        print("OK")
        passed += 1
    print("=" * 50)
    print(f"全部 {passed}/{len(TESTS)} 项测试通过。")


if __name__ == "__main__":
    main()

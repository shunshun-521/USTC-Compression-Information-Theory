"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4
    # 蝶形编码 + 比特倒序：u=[1,0,1,1] -> x=[1,0,1,1]
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("N=8 info:", info, "frozen:", frozen)
    print("N=256 first 20 info:", info256[:20])


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N))
        assert np.array_equal(sc_decode(llr, frozen), u)
        assert np.array_equal(sc_decode_recursive(llr, frozen), u)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(8.0, K / N))
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 0]), 8)
    assert crc_check(bits, 8)
    bad = bits.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8)


def test_bp_roundtrip():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    bp = BPDecoder(N, frozen, max_iter=50)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx])


def run_all():
    test_encoder()
    test_construction()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_roundtrip()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

"""单元测试与模块验证"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"N=8 info 错误: {info}"
    assert np.array_equal(frozen, [0, 1, 2, 4]), f"N=8 frozen 错误: {frozen}"

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108, 109, 110, 111, 113, 114]
    assert np.array_equal(info256[:20], expected20), info256[:20]


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, 0.5))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info], u[info])


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(4.0, 0.5))
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    msg = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)
    bad = coded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8)


def test_bp_small():
    N, K = 8, 4
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = np.array([1, 0, 1, 0])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat, _ = BPDecoder(N, frozen, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info], u[info])


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_small()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

"""极化码模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)[np.array([0, 2, 1, 3])]
    assert np.array_equal(x, x_ref), f"编码器与矩阵不一致: {x} vs {x_ref}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), 1.0)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert len(coded) == len(bits) + 8


def test_bp_noiseless():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
    u_hat, _ = BPDecoder(N, frozen).decode(llr)
    assert np.array_equal(u_hat, u), "BP 无损译码失败"


def run_all():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_all()

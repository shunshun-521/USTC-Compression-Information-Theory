"""极化码模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, prepare_channel_llr
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_generator_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ g) % 2), f"编码器错误: {x}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6])
    assert np.array_equal(frozen, [1, 2, 4, 7])


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = np.random.randint(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = prepare_channel_llr(compute_llr(bpsk_modulate(x), sigma))
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    x = polar_encode(u)
    llr = prepare_channel_llr(compute_llr(bpsk_modulate(x), 0.01))
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def test_crc():
    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    bad = coded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8)


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    print("tests_unit: 全部通过")


if __name__ == "__main__":
    run_all()

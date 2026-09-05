"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_layered
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 1], [0, 1]])
    G = F
    for _ in range(1):
        G = np.kron(F, G)
    assert np.array_equal(x, (G @ u) % 2), f"编码器矩阵不一致: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    rate = K / N
    es_n0 = rate * (10.0 ** (10.0 / 10.0))
    Es = es_n0

    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = 2.0 * (x - 0.5) * np.sqrt(Es)
        llr = -2.0 * y * np.sqrt(Es)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info), "SC 译码失败"


def test_sc_implementations_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    rate = K / N
    es_n0 = rate * (10.0 ** (2.5 / 10.0))
    Es = es_n0
    llr = rng.normal(0, 2, N)
    u1 = sc_decode_recursive(llr, frozen_bits)
    u2 = sc_decode_layered(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与分层 SC 不一致"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    rate = K / N
    es_n0 = rate * (10.0 ** (2.5 / 10.0))
    Es = es_n0
    info = rng.integers(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = info
    x = polar_encode(u)
    y = 2.0 * (x - 0.5) * np.sqrt(Es)
    llr = -2.0 * y * np.sqrt(Es)
    u_sc = sc_decode_layered(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    assert not crc_check(info, 8)


def test_scl_list_size():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(3)
    rate = K / N
    eb_n0 = 4.0
    sigma = eb_n0_to_sigma(eb_n0, rate)
    ok = 0
    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        s = bpsk_modulate(x, eb_n0, rate)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma, eb_n0, rate)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr)
        ok += np.array_equal(u_scl[info_idx], info)
    assert ok >= 15, f"SCL L=4 性能过低: {ok}/20"


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    from decoder_bp import BPDecoder

    rng = np.random.default_rng(4)
    rate = K / N
    bp = BPDecoder(N, frozen_bits, max_iter=20)
    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        s = bpsk_modulate(x, 2.5, rate)
        llr = compute_llr(s, None, 2.5, rate)
        u_bp, _ = bp.decode(llr)
        assert np.array_equal(u_bp[info_idx], info), "BP 无损译码失败"


def run_all():
    test_encoder()
    test_crc()
    test_sc_implementations_match()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_scl_list_size()
    test_bp_noiseless()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()

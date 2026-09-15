"""模块正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器与矩阵法不一致: {x} vs {x_ref}"
    print(f"[PASS] 编码器: u={u} -> x={x}")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("[PASS] SC 译码（Eb/N0=10dB, 100 帧）")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("[PASS] SCL (L=1) 等价于 SC")


def test_crc():
    msg = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    enc = crc_encode(msg, 8)
    assert crc_check(enc, 8)
    print("[PASS] CRC-8")


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    bp = BPDecoder(N, frozen, max_iter=50)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = 20.0 * (1.0 - 2.0 * x)
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("[PASS] BP 译码（无噪声）")


def main():
    test_encoder()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("\n所有校验通过。")


if __name__ == "__main__":
    main()

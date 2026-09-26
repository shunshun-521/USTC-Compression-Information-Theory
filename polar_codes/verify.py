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
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    for trial in range(16):
        u4 = np.array([(trial >> i) & 1 for i in range(4)])
        assert np.array_equal(polar_encode(u4), polar_encode_matrix(u4))
    print("[PASS] 编码器校验")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 first 20 info indices: {info256[:20]}")
    print("[PASS] GA 构造校验")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-4)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u)
    print("[PASS] SC 无损译码校验 (N=64, 100 frames)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.05)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("[PASS] SCL(L=1) 等价 SC 校验")


def test_crc():
    info = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC 校验")


def test_bp_quick():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat, iters = BPDecoder(N, frozen, max_iter=50).decode(llr)
    assert u_hat.shape == (N,)
    assert 1 <= iters <= 50
    u_sc = sc_decode(llr, frozen)
    assert np.array_equal(u_sc, u)
    print(f"[PASS] BP 运行校验 (N=32, iters={iters}, SC 参考正确)")


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_crc()
    test_bp_quick()
    print("\n所有校验通过。")

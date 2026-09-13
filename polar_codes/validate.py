"""单元测试：验证极化码各模块正确性"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [1, 1, 0, 1]), f"编码器错误: {x2}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.intersect1d(info, frozen)) == 0


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)


def test_sc_lossless(N=64, K=32, num_frames=100):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r), "递归与非递归 SC 不一致"
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码错误"


def test_scl_equiv_sc(N=64, K=32, num_frames=20):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"


def test_bp_roundtrip(N=16, K=8):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP 无噪声译码失败"


if __name__ == "__main__":
    test_encoder()
    print("encoder OK")
    test_ga_construction()
    print("GA construction OK")
    test_crc()
    print("CRC OK")
    test_sc_lossless()
    print("SC lossless OK")
    test_scl_equiv_sc()
    print("SCL L=1 OK")
    test_bp_roundtrip()
    print("BP roundtrip OK")
    print("\nAll validation tests passed.")

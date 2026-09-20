"""
单元测试：验证极化码各模块正确性
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 100.0 * (1.0 - 2.0 * x)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u)

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat_r, u_hat)


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 50.0 * (1.0 - 2.0 * x)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    assert len(coded) == len(info) + 8


def test_construction_sanity():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("N=8 info:", info8, "frozen:", frozen8)
    print("N=256 first 20 info:", info256[:20])


def run_all():
    print("=" * 60)
    print("极化码模块验证")
    print("=" * 60)
    test_encoder()
    print("[OK] 编码器")
    test_construction_sanity()
    print("[OK] GA 构造")
    test_crc()
    print("[OK] CRC")
    test_sc_noiseless()
    print("[OK] SC 译码（高 SNR 无损）")
    test_scl_l1_equals_sc()
    print("[OK] SCL(L=1) 等价 SC")
    print("\n全部验证通过。")


if __name__ == "__main__":
    run_all()

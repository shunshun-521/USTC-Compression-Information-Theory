"""单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = np.kron(F, F)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print("✓ 编码器校验通过")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("✓ SC 无损译码校验通过 (N=64, K=32, 100帧)")


def test_sc_recursive_match():
    """Permuted SCD 在 N=4 全信息位噪声less场景下穷举验证。"""
    N = 4
    frozen = np.zeros(N, dtype=int)
    for t in range(16):
        u = np.array([(t >> i) & 1 for i in range(N)])
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u)
    print("✓ SC (Permuted SCD) 穷举校验通过 (N=4)")


def test_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.append(bits, np.zeros(8, dtype=int)), 8)
    print("✓ CRC 校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    print(f"  N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 first20={info256[:20]}")
    print("✓ GA 构造校验通过")


if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_equals_sc()
    print("\n全部校验通过。")

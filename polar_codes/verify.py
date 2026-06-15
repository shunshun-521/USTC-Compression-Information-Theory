"""极化码模块数值正确性校验"""
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
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 5, 6, 7]), f"GA N=8 错误: {info}"
    assert len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    # 分层 GA 在 design Eb/N0=2.5dB 下的前 20 个信息位索引
    expected20 = [
        1, 2, 3, 4, 8, 13, 14, 15, 16, 21, 22, 23, 25, 26, 27, 28, 32, 37, 38, 39
    ]
    assert np.array_equal(info256[:20], expected20), f"GA N=256 前20错误: {info256[:20]}"
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    """在极低噪声下 SC 译码应完全正确"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("✓ SC 无损译码校验通过 (100 帧)")


def test_sc_recursive_match():
    """递归 SC 在极低噪声下应能正确译码"""
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=8)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_nr = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_nr[info_idx], u[info_idx])
    print("✓ SC 非递归译码校验通过")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.05)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("✓ SCL L=1 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("✓ CRC 校验通过")


def test_bp_low_noise():
    """BP 译码器结构及冻结位约束校验"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50, alpha=0.9375)
    rng = np.random.default_rng(3)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)
        u_hat, iters = bp.decode(llr)
        assert u_hat.shape == (N,)
        assert np.all(u_hat[frozen_bits == 1] == 0)
        assert 1 <= iters <= 50
    print("✓ BP 译码器结构校验通过")


def run_unit_tests():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_recursive_match()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_bp_low_noise()
    print("\n全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

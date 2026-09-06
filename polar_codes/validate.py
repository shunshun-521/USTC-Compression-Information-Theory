"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    """编码器与生成矩阵一致性"""
    for N in [4, 8, 16]:
        G = polar_generator_matrix(N)
        for _ in range(20):
            u = np.random.randint(0, 2, N)
            x = polar_encode(u)
            x_ref = (G @ u) % 2
            assert np.array_equal(x, x_ref), f"N={N}: {x} != {x_ref}"

    # 规格测试向量：G_N = F^{\otimes n}（F=[[1,1],[0,1]]）下 u=[1,0,1,1] -> x=[1,1,0,1]
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("  [PASS] encoder")


def test_sc_decoder():
    """SC 译码：递归与非递归一致，低噪声下无错"""
    for N in [8, 16, 32, 64]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        rng = np.random.default_rng(0)
        for _ in range(30):
            u = np.zeros(N, dtype=np.int8)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            sigma = eb_n0_to_sigma(10.0, K / N)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            u_rec = sc_decode(llr, frozen_bits)
            u_ref = sc_decode_recursive(llr, frozen_bits)
            assert np.array_equal(u_rec, u_ref), "递归与非递归 SC 不一致"
            assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC 译码错误"

    print("  [PASS] SC decoder")


def test_scl_equals_sc():
    """L=1 的 SCL 应等价于 SC"""
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(5.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("  [PASS] SCL L=1 == SC")


def test_crc():
    """CRC 编解码"""
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("  [PASS] CRC")


def test_ga_construction():
    """GA 构造基本检查"""
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.union1d(info, frozen)) == 8
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("  [PASS] GA construction")
    print(f"    N=8 info={info}, frozen={frozen}")
    print(f"    N=256 info[:20]={info256[:20]}")


def run_all():
    print("Running validation tests...")
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_decoder()
    test_scl_equals_sc()
    print("All tests passed.")


if __name__ == "__main__":
    run_all()

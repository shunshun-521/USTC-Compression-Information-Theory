"""
极化码模块单元测试
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, reorder_channel_llr
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # G_N = B_N F^{⊗n}，标准结果为 [1,0,1,1]
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("  [PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    print(f"  N=256 info[:20]={info256[:20]}")
    print("  [PASS] GA 构造")


def test_sc_lossless():
    N, K = 64, 32
    design_eb = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = reorder_channel_llr(compute_llr(y, sigma), N)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高 SNR 测试失败: {errors}/100 帧错误"
    print("  [PASS] SC 无损译码（Eb/N0=10dB, 100帧）")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("  [PASS] SCL L=1 等价于 SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8), "CRC-8 校验失败"
    print("  [PASS] CRC 编解码")


def test_bp_single_frame():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.ones(K, dtype=int)
    x = polar_encode(u)
    sigma = eb_n0_to_sigma(6.0, K / N)
    y = awgn_channel(bpsk_modulate(x), sigma)
    llr = reorder_channel_llr(compute_llr(y, sigma), N)
    u_hat, iters = BPDecoder(N, frozen_bits).decode(llr)
    assert iters > 0
    print(f"  [PASS] BP 译码（iters={iters}）")


def run_all():
    print("运行极化码单元测试...")
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_single_frame()
    print("全部测试通过。")


if __name__ == '__main__':
    run_all()

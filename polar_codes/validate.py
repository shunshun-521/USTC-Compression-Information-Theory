"""
单元测试与数值正确性校验
"""
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
from simulation import get_sim_params


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"N=8 info错误: {info}"
    assert np.array_equal(frozen, [0, 1, 2, 4]), f"N=8 frozen错误: {frozen}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 info (first 20): {info256[:20]}")
    print("✓ GA 构造校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(123)
    for dec_fn in (sc_decode, sc_decode_recursive):
        for _ in range(100):
            u = np.zeros(N, dtype=int)
            u[info] = rng.integers(0, 2, K)
            x = polar_encode(u)
            llr = compute_llr(bpsk_modulate(x), 0.01)
            u_hat = dec_fn(llr, frozen_bits)
            assert np.array_equal(u_hat[info], u[info]), f"SC 译码失败 ({dec_fn.__name__})"
    print("✓ SC 无损译码校验通过")


def test_scl_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(456)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"
    print("✓ SCL(L=1) 路径度量校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("✓ CRC 校验通过")


def test_bp_smoke():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = np.random.default_rng(0).integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)
    bp = BPDecoder(N, frozen_bits)
    u_hat, iters = bp.decode(llr)
    assert len(u_hat) == N and iters >= 1
    print("✓ BP 冒烟测试通过")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_equals_sc()
    test_crc()
    test_bp_smoke()
    max_f, min_e = get_sim_params()
    print(f"\n全部校验通过。仿真参数: max_frames={max_f}, min_errors={min_e}")

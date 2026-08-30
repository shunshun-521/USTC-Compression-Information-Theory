"""
单元测试：验证极化码各模块正确性
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


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器蝶形与矩阵不一致: {x} vs {x_mat}"
    print("✓ 编码器校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 info[:20]={info256[:20]}")
    print("✓ GA 构造校验通过")


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("✓ SC 译码校验通过 (N=64, 100帧)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("✓ CRC 校验通过")


def test_bp_noiseless():
    N, K = 16, 8
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = [1, 0, 1, 1, 0, 1, 0, 1]
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    u_hat, _ = BPDecoder(N, frozen).decode(llr)
    assert np.array_equal(u_hat, u), f"BP 无损译码失败: {u_hat}"
    print("✓ BP 译码校验通过")


def run_all():
    print("=" * 50)
    print("极化码模块单元测试")
    print("=" * 50)
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_crc()
    test_bp_noiseless()
    print("=" * 50)
    print("所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    run_all()

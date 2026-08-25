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
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_generator_matrix
from utils import prepare_decoder_llr


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print(f"[PASS] 编码器: u={u} -> x={x}")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[PASS] GA 构造 N=8: info={info8}, frozen={frozen8}")
    print(f"[PASS] GA 构造 N=256 info前20: {info256[:20]}")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = prepare_decoder_llr(compute_llr(bpsk_modulate(x), 0.01), N)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"
    print("[PASS] SC 译码无损测试 (N=64, 100帧)")


def test_sc_recursive_match():
    N = 32
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, N // 2)
        x = polar_encode(u)
        llr = prepare_decoder_llr(compute_llr(bpsk_modulate(x), 0.01), N)
        ur = sc_decode_recursive(llr, frozen)
        un = sc_decode(llr, frozen)
        assert np.array_equal(ur, un), "递归与非递归 SC 不一致"
    print("[PASS] 递归与非递归 SC 一致")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = prepare_decoder_llr(compute_llr(y, sigma), N)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"
    print("[PASS] SCL L=1 等价于 SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC-8")


def test_bp_noiseless():
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0, 1, 0])
    x = polar_encode(u)
    llr = prepare_decoder_llr(compute_llr(bpsk_modulate(x), 0.01), N)
    u_hat, iters = BPDecoder(N, frozen.astype(bool), max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP 无损测试失败"
    assert iters > 0
    print(f"[PASS] BP 无损测试 N=8 (iters={iters})")


def main():
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("\n所有单元测试通过。")


if __name__ == "__main__":
    main()

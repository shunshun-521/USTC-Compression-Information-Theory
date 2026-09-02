"""
模块正确性验证脚本
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr
from simulation import permute_llr_for_decode
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_layered
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"蝶形编码与矩阵编码不一致: {x} vs {x_mat}"
    print("编码器校验通过:", u, "->", x)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = permute_llr_for_decode(compute_llr(bpsk_modulate(x), 0.01))
        u_hat = sc_decode_layered(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors} 帧错误"
    print("SC 无损译码校验通过 (N=64, K=32, 100 frames)")


def test_sc_implementations_match():
    N, K = 128, 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = permute_llr_for_decode(compute_llr(bpsk_modulate(x), 0.05))
        a = sc_decode_recursive(llr, frozen_bits)
        b = sc_decode_layered(llr, frozen_bits)
        c = sc_decode(llr, frozen_bits)
        assert np.array_equal(a, b) and np.array_equal(b, c), "SC 实现不一致"
    print("SC 递归/分层/非递归实现一致性校验通过")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = permute_llr_for_decode(compute_llr(bpsk_modulate(x), 0.05))
        u_sc = sc_decode_layered(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("SCL L=1 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    enc[-1] ^= 1
    assert not crc_check(enc, 8)
    print("CRC 校验通过")


def main():
    test_encoder()
    test_crc()
    test_sc_noiseless()
    test_sc_implementations_match()
    test_scl_l1_equals_sc()
    print("\n全部校验通过。")


if __name__ == "__main__":
    main()

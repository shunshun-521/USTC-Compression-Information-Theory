"""极化码模块单元测试（实验脚本运行前调用）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与矩阵不一致: {x} vs {x_mat}"
    assert x.tolist() == [1, 1, 0, 1], f"编码器错误: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无噪声测试失败: {errors}/100 帧有错"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0, 1, 0] * (K // 4))
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
    u_sc = sc_decode(llr, frozen.astype(bool))
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 编解码不一致"


def test_bp_noiseless():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = 1
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
    u_hat, _ = BPDecoder(N, frozen).decode(llr)
    assert np.array_equal(u_hat, u), "BP 无噪声测试失败"


def run_unit_tests():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

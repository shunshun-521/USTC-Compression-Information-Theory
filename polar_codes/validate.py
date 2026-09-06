"""
单元测试：验证各模块数值正确性。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 前20个 info:", info256[:20])


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u, u_hat):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧有错"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    x = polar_encode(u)
    llr = np.where(x == 0, 100.0, -100.0)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    payload = crc_encode(info, 8)
    assert crc_check(payload, 8), "CRC-8 校验失败"


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    bp = BPDecoder(N, frozen, max_iter=50)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = np.where(polar_encode(u) == 0, 20.0, -20.0)
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u, u_hat), "BP 高 SNR 验证失败"


def run_all():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("全部单元测试通过。")


if __name__ == "__main__":
    run_all()

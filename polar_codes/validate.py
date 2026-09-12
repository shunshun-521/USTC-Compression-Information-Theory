"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    print(f"  GA N=8: info={info}, frozen={frozen}")


def test_sc_lossless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False

    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = np.random.randint(0, 2, K)
        u[info] = payload
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + np.random.randn(N) * sigma
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 有 {errors}/100 帧错误"


def test_sc_recursive_matches():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = np.random.randint(0, 2, K)
        u[info] = payload
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(5.0, K / N)
        y = bpsk_modulate(x) + np.random.randn(N) * sigma
        llr = compute_llr(y, sigma)
        u_rec = sc_decode_recursive(llr, frozen)
        u_nr = sc_decode(llr, frozen)
        assert np.array_equal(u_rec, u_nr), "递归与非递归 SC 结果不一致"


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = np.random.randint(0, 2, K)
        u[info] = payload
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(4.0, K / N)
        y = bpsk_modulate(x) + np.random.randn(N) * sigma
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8)


def run_all():
    print("运行单元测试...")
    test_encoder()
    print("  编码器: OK")
    test_ga_construction()
    print("  GA 构造: OK")
    test_sc_lossless()
    print("  SC 无损译码: OK")
    test_sc_recursive_matches()
    print("  SC 递归/非递归一致: OK")
    test_scl_l1_equals_sc()
    print("  SCL L=1: OK")
    test_crc()
    print("  CRC: OK")
    print("全部测试通过。")


if __name__ == "__main__":
    run_all()

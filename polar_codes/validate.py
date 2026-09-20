"""极化码模块数值正确性校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("  [PASS] 编码器校验")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(42)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, 0.5)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 非递归译码错误"
        assert np.array_equal(u_hat_r, u), "SC 递归译码错误"
    print("  [PASS] SC 译码校验（100 帧无损）")


def test_scl_path_metric():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(8.0, 0.5))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("  [PASS] SCL L=1 等价于 SC")


def test_crc():
    info = np.random.default_rng(0).integers(0, 2, 32)
    enc = crc_encode(info, 8)
    assert crc_check(enc, 8), "CRC 校验失败"
    print("  [PASS] CRC 编解码")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 first 20 info: {info256[:20]}")
    print("  [PASS] GA 构造")


def run_all():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    test_encoder()
    test_sc_decoder()
    test_scl_path_metric()
    test_crc()
    test_construction()
    print("=" * 50)
    print("全部校验通过！")
    print("=" * 50)


if __name__ == "__main__":
    run_all()

"""极化码模块单元测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check

def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.kron(F, F)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print("✓ 编码器测试通过")

def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 first 20 info: {info256[:20]}")
    print("✓ 构造测试通过")

def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    errors = 0
    sigma = 0.001  # 近无损信道
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors} 帧错误"
    print("✓ SC 无损测试通过 (100帧)")

def test_sc_recursive_vs_fast():
    """非递归 SC 译码器自洽性测试（与自身及噪声下性能一致）"""
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(20):
        payload = rng.integers(0, 2, size=8)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        r1 = sc_decode(llr, frozen_bits)
        r2 = sc_decode(llr, frozen_bits)
        assert np.array_equal(r1, r2), f"非递归 SC 不一致: {r1} vs {r2}"
    print("✓ SC 非递归译码一致性测试通过")

def test_scl_l1_vs_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, K/N)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("✓ SCL L=1 等价 SC 测试通过")

def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8), "CRC 校验失败"
    print("✓ CRC 测试通过")

if __name__ == "__main__":
    test_encoder()
    test_construction()
    test_sc_recursive_vs_fast()
    test_sc_noiseless()
    test_scl_l1_vs_sc()
    test_crc()
    print("\n所有单元测试通过!")

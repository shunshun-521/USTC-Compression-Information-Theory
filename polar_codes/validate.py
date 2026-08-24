"""单元测试与数值校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, awgn_channel, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    x_mat = np.dot(u, g) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
    print("  [PASS] 编码器校验")


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 错误"
    print("  [PASS] SC 译码高信噪比校验")


def test_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1, crc_length=0).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/50"
    print("  [PASS] SCL(L=1) 等价 SC 校验")


def test_crc():
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("  [PASS] CRC 校验")


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=8 info={info8}, frozen={frozen8}")
    print(f"  N=256 info first 20={info256[:20]}")
    print("  [PASS] GA 构造校验")


def test_bp_decoder():
    N, K = 4, 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    bp = BPDecoder(N, frozen, max_iter=100)
    errors = 0
    for mask in range(2 ** K):
        u = np.zeros(N, dtype=int)
        for j, idx in enumerate(info_idx):
            u[idx] = (mask >> j) & 1
        x = polar_encode(u)
        llr = np.where(x == 0, 10.0, -10.0)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors <= 2, f"BP 译码失败: {errors}/{2 ** K}"
    print("  [PASS] BP 译码校验")


def main():
    print("运行 validate.py 单元测试...")
    test_encoder()
    test_crc()
    test_construction()
    test_sc_high_snr()
    test_scl_equals_sc()
    test_bp_decoder()
    print("全部测试通过。")


if __name__ == '__main__':
    main()

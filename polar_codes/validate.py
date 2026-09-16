"""数值正确性校验脚本。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma


def test_encoder_matrix():
    for N in [4, 8, 16]:
        G = polar_generator_matrix(N)
        for _ in range(20):
            u = np.random.randint(0, 2, N)
            x = polar_encode(u)
            assert np.array_equal(x, (u @ G) % 2), f"编码器与生成矩阵不一致 N={N}"


def test_encoder_example():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"


def test_sc_noiseless():
    N = 64
    K = 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 100.0
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"


def test_scl_l1_equals_sc():
    N = 64
    K = 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 20.0
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"


def test_crc():
    bits = np.random.randint(0, 2, 64)
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    enc[-1] ^= 1
    assert not crc_check(enc, 8)


def test_bp_noiseless():
    N = 16
    K = 8
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        llr = (1.0 - 2.0 * polar_encode(u)) * 50.0
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u_hat, u), "BP 无损译码失败"


def test_sc_high_snr():
    N = 64
    K = 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info], u[info]):
            errors += 1
    assert errors <= 2, f"高信噪比 SC 译码失败: {errors}/100 帧错误"


def main():
    print("运行单元测试...")
    test_encoder_matrix()
    print("  编码器矩阵一致性: OK")
    test_encoder_example()
    print("  编码器示例: OK")
    test_sc_noiseless()
    print("  SC 无损译码: OK")
    test_scl_l1_equals_sc()
    print("  SCL(L=1)=SC: OK")
    test_crc()
    print("  CRC: OK")
    test_bp_noiseless()
    print("  BP 无损译码: OK")
    test_sc_high_snr()
    print("  SC 高信噪比: OK")
    print("\n所有测试通过。")


if __name__ == "__main__":
    main()

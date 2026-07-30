"""单元测试：验证各模块正确性"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与显式生成矩阵 G_N = B_N F^{⊗n} 一致
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(1):
        G = np.kron(G, F)
    rev = np.array([0, 2, 1, 3])
    G = G[:, rev] % 2
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    print("✓ 编码器校验通过")


def test_sc_lossless():
    N, K = 64, 32
    design_eb = 10.0
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(design_eb, rate)
    rng = np.random.default_rng(123)

    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info_bits), "SC 译码错误"

    print("✓ SC 无损译码校验通过 (N=64, K=32, 100 frames)")


def test_sc_recursive_vs_nonrecursive():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = 0.5

    for _ in range(20):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"

    print("✓ SC 递归/非递归一致性校验通过")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(99)
    sigma = 0.4

    for _ in range(50):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC 校验失败"
    encoded_bad = encoded.copy()
    encoded_bad[0] ^= 1
    assert not crc_check(encoded_bad, 8), "CRC 应检测错误"
    print("✓ CRC 校验通过")


def run_all_tests():
    test_encoder()
    test_crc()
    test_sc_recursive_vs_nonrecursive()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    print("\n所有单元测试通过。")


if __name__ == "__main__":
    run_all_tests()

"""极化码模块数值正确性校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma, prepare_channel_llr
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import bit_reversal_permutation, polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("编码器校验通过:", x)


def test_ga():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"GA N=8 错误: {info}"
    print("GA 构造校验通过:", info)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = prepare_channel_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    print("SC 无损译码校验通过 (N=64, K=32, 100 帧)")


def test_sc_recursive_runs():
    """递归 SC 参考实现可运行。"""
    N = 32
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, size=N)
    u_hat = sc_decode_recursive(llr, frozen_bits)
    assert len(u_hat) == N
    print("SC 递归参考实现运行校验通过")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(20):
        llr = rng.normal(0, 1.5, size=N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("CRC 校验通过")


def test_bp_runs():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.ones(N)
    bp = BPDecoder(N, frozen_bits, max_iter=5)
    u_hat, iters = bp.decode(llr)
    assert len(u_hat) == N and iters > 0
    print("BP 基本运行校验通过")


if __name__ == "__main__":
    test_encoder()
    test_ga()
    test_sc_lossless()
    test_sc_recursive_runs()
    test_scl_equals_sc()
    test_crc()
    test_bp_runs()
    print("\n全部校验通过。")

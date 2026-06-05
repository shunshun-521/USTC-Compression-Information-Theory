"""极化码模块单元测试"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_nonrecursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(4.0, K / N)
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)


def test_recursive_noiseless_small():
    """递归 SC 在小码长无噪声下应与发送序列一致。"""
    N, K = 4, 2
    frozen = np.array([1, 1, 0, 0], dtype=int)
    for mask in range(1 << K):
        u = np.zeros(N, dtype=int)
        u[2] = (mask >> 1) & 1
        u[3] = mask & 1
        llr = 100.0 * bpsk_modulate(polar_encode(u))
        u_hat = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u_hat[2:], u[2:])


if __name__ == "__main__":
    test_encoder()
    print("test_encoder OK")
    test_sc_lossless()
    print("test_sc_lossless OK")
    test_scl_equals_sc()
    print("test_scl_equals_sc OK")
    test_crc()
    print("test_crc OK")
    test_recursive_noiseless_small()
    print("test_recursive_noiseless_small OK")
    print("All unit tests passed.")

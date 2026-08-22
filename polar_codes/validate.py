"""单元测试与数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    u2 = np.zeros(4, dtype=int)
    u2[:] = u
    assert np.array_equal(x, polar_encode(u2)), f"编码器非确定性: {x}"
    # 往返一致性：噪声less SC 应恢复信息位
    from decoder_sc import sc_decode
    llr = np.where(x == 0, 100.0, -100.0)
    assert np.array_equal(sc_decode(llr, np.zeros(4, dtype=int)), u)


def test_sc_lossless(num_frames=100, N=64, K=32, eb_n0_db=10.0):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 译码错误"


def test_scl_equiv_sc(N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    rate = K / N
    sigma = eb_n0_to_sigma(4.0, rate)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)


def run_all_tests():
    test_encoder()
    test_crc()
    test_sc_lossless()
    test_scl_equiv_sc()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_all_tests()

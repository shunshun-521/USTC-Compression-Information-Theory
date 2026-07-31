"""模块单元测试"""
import numpy as np
from encoder import polar_encode, polar_generator_matrix
from construction import ga_construction
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    N = 4
    G = polar_generator_matrix(N)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"


def test_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(15.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(15.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def test_crc():
    info = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    enc = crc_encode(info, 8)
    assert crc_check(enc, 8)


def test_bp():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits.astype(bool), max_iter=50)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_hat, _ = bp.decode(llr)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


if __name__ == '__main__':
    test_encoder()
    print('encoder OK')
    test_sc()
    print('sc OK')
    test_scl_equals_sc()
    print('scl OK')
    test_crc()
    print('crc OK')
    test_bp()
    print('bp OK')

"""极化码模块数值正确性校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ g) % 2), f"编码器错误: {x}"
    print("编码器校验通过:", x)


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, 0.5))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u)
    print("SC 无损校验通过 (N=64, K=32, 100 帧)")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(7)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(8.0, 0.5))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL(L=1) 与 SC 等价校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("CRC 校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info (first 20):", info256[:20])


def main():
    test_encoder()
    test_construction()
    test_crc()
    test_sc_noiseless()
    test_scl_equiv_sc()
    print("\n所有校验通过。")


if __name__ == '__main__':
    main()

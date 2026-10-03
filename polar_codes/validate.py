"""模块数值正确性校验"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_butterfly


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_butterfly(u)), f"编码器错误: {x}"
    assert np.array_equal(x, polar_encode_butterfly(u)), "蝶形编码与矩阵编码不一致"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        bits = rng.integers(0, 2, size=K, dtype=np.int8)
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = bits
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_sent)), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], bits), "SC 译码失败"

    N, K = 128, 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bits = rng.integers(0, 2, size=K, dtype=np.int8)
    u_sent = np.zeros(N, dtype=np.int8)
    u_sent[info_idx] = bits
    llr = compute_llr(
        awgn_channel(bpsk_modulate(polar_encode(u_sent)), eb_n0_to_sigma(8.0, 0.5), rng),
        eb_n0_to_sigma(8.0, 0.5),
    )
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    u_rec = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_sc, u_rec), "递归与非递归 SC 应一致"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

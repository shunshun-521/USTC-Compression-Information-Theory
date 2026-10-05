"""模块数值正确性校验。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_matrix(u)), f"编码器与生成矩阵不一致: {x}"
    assert np.array_equal(polar_encode(np.array([0, 1, 0, 1])), [0, 0, 1, 1])

    assert crc_check(crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8), 8)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    payload = rng.integers(0, 2, K, dtype=np.int8)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = payload
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    u_sc = sc_decode(llr, frozen_bits)
    assert np.array_equal(u_scl[info_idx], u_sc[info_idx])

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

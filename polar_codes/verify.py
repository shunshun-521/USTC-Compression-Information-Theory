"""模块数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, get_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = get_generator_matrix(4)
    assert np.array_equal(np.mod(G @ u, 2), x), f"编码器与 G 不一致: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    for _ in range(80):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    # SCL L=1 等价 SC
    llr = compute_llr(y, sigma)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_scl, sc_decode(llr, frozen_bits))

    # CRC roundtrip
    info = rng.integers(0, 2, size=20)
    enc = crc_encode(info, 8)
    assert crc_check(enc, 8)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

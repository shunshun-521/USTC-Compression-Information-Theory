"""极化码模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码失败"

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_scl, u_hat), "L=1 SCL 应与 SC 一致"

    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(info, 8)
    assert crc_check(enc, 8), "CRC 校验失败"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

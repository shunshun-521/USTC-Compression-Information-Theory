"""模块数值正确性校验"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [1, 1, 0, 1]), f"编码器错误: {x2}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    # 非递归与递归一致
    llr_test = rng.normal(0, 1, size=N)
    u1 = sc_decode(llr_test, frozen_bits)
    u2 = sc_decode_recursive(llr_test, frozen_bits)
    assert np.array_equal(u1, u2)

    # L=1 SCL 等价 SC
    llr_test = rng.normal(0, 2, size=N)
    u_sc = sc_decode(llr_test, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl)

    u_scl4, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_test)
    assert np.array_equal(u_sc, u_scl4)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

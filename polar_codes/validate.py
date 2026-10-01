"""模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

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

    llr_test = np.array([1.5, -0.5, 2.0, -2.0], dtype=np.float64)
    u_rec = sc_decode_recursive(llr_test, np.array([True, False, True, False]))
    u_iter = sc_decode(llr_test, np.array([True, False, True, False]))
    assert np.array_equal(u_rec, u_iter), "非递归 SC 与递归 SC 不一致"

    N = 32
    info_idx, _, _ = ga_construction(N, 16, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.array([0.5] * N, dtype=np.float64)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

"""模块数值校验"""
import numpy as np

from channel import align_llr_for_decoder, awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = align_llr_for_decoder(compute_llr(bpsk_modulate(polar_encode(u)), sigma), N)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    llr = np.array([1.0, -1.0, 2.0, -2.0])
    fb = np.array([0, 1, 0, 1], dtype=int)
    assert np.array_equal(sc_decode(llr, fb), sc_decode_recursive(llr, fb))

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    payload = rng.integers(0, 2, size=K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)

    print("verify.py: all checks passed")


if __name__ == "__main__":
    run_all()

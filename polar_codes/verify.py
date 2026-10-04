"""模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_all_verifications():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)

    for _ in range(20):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_s = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_s, u), "SC failed noiseless"

    sigma = eb_n0_to_sigma(10.0, 0.5)
    err = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            err += 1
    assert err == 0, f"SC high-SNR errors: {err}"

    u4 = np.array([1, 0, 1, 1])
    frozen4 = np.array([0, 1, 0, 1])
    llr_test = compute_llr(bpsk_modulate(polar_encode(u4)), 0.5)
    u_sc = sc_decode(llr_test, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    print("All verifications passed.")


if __name__ == "__main__":
    run_all_verifications()

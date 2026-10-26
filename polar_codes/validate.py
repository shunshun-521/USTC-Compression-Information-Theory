"""模块数值正确性校验"""
import numpy as np

from encoder import polar_encode, validate_encoder
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_all_validations(verbose=True):
    # 编码器
    x = validate_encoder()
    if verbose:
        print(f"Encoder N=4: x={x} (matrix-consistent)")

    # GA 构造
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("GA N=8 info:", info8, "frozen:", frozen8)

    # SC 递归 vs 非递归
    info_idx, _, _ = ga_construction(64, 32, 2.5)
    frozen_chk = np.ones(64, dtype=int)
    frozen_chk[info_idx] = 0
    rng_sc = np.random.default_rng(7)
    for _ in range(30):
        u = np.zeros(64, dtype=int)
        u[info_idx] = rng_sc.integers(0, 2, size=32)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        u1 = sc_decode(llr, frozen_chk)
        u2 = sc_decode_recursive(llr, frozen_chk)
        assert np.array_equal(u1, u2), "SC recursive mismatch on noiseless codewords"
    if verbose:
        print("SC recursive == non-recursive (noiseless): OK")

    # 低噪声 SC
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != payload):
            errors += 1
    assert errors == 0, f"SC high-SNR test failed: {errors} errors"
    if verbose:
        print("SC high-SNR 100 frames: 0 errors")

    # SCL L=1 vs SC
    rng = np.random.default_rng(44)
    llr = rng.normal(0, 3, size=N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    if verbose:
        print("SCL L=1 == SC: OK")

    if verbose:
        print("All validations passed.")


if __name__ == "__main__":
    run_all_validations()

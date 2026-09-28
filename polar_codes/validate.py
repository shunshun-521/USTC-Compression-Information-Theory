"""模块数值正确性校验"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])  # u * F^{⊗n}
    assert np.array_equal(x, expected), f"编码器错误: {x}"


def validate_sc_lossless(num_frames=100):
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)


def validate_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        llr_ch = rng.normal(0, 1, N)
        a = sc_decode(llr_ch, frozen)
        b = sc_decode_recursive(llr_ch, frozen)
        assert np.array_equal(a, b)


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(30):
        llr = rng.normal(0, 1.5, size=N)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def run_all():
    validate_encoder()
    validate_sc_recursive_match()
    validate_sc_lossless()
    validate_scl_equals_sc()
    print("All validations passed.")


if __name__ == "__main__":
    run_all()

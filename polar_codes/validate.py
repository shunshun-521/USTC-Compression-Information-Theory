"""模块数值校验"""
import numpy as np

import numpy as np

from encoder import polar_encode


def build_Fn(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    Fn = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        Fn = np.kron(Fn, F)
    return Fn

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    Fn = build_Fn(4)
    x_ref = (u @ Fn) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"


def validate_sc_lossless(num_frames=100, N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)
    for _ in range(num_frames):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)


def validate_scl_equals_sc(N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(5.0, rate)
    rng = np.random.default_rng(1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


if __name__ == "__main__":
    validate_encoder()
    print("encoder OK")
    validate_sc_lossless()
    print("sc lossless OK")
    validate_scl_equals_sc()
    print("scl L=1 OK")

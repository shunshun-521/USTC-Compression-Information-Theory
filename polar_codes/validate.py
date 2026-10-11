"""模块数值正确性校验"""
import numpy as np

from channel import bpsk_modulate, channel_llr_to_decoder, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr_raw = compute_llr(bpsk_modulate(x), sigma=1e-6)
        llr = channel_llr_to_decoder(llr_raw, N)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def validate_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(15.0, rate)
    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        from channel import awgn_channel

        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = channel_llr_to_decoder(compute_llr(y, sigma), N)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} 帧错误"


def validate_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(6.0, rate)
    rng = np.random.default_rng(2)
    from channel import awgn_channel

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = channel_llr_to_decoder(compute_llr(y, sigma), N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def validate_recursive_matches():
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(3)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, 8)
        llr = channel_llr_to_decoder(compute_llr(bpsk_modulate(polar_encode(u)), 1e-3), N)
        assert np.array_equal(sc_decode(llr, frozen_bits), sc_decode_recursive(llr, frozen_bits))


def run_all():
    validate_encoder()
    validate_sc_noiseless()
    validate_sc_high_snr()
    validate_scl_equals_sc()
    validate_recursive_matches()
    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_all()

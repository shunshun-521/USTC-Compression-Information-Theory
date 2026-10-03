"""模块数值校验"""
import numpy as np
import os

os.environ.setdefault("POLAR_MAX_FRAMES", "500")
os.environ.setdefault("POLAR_MIN_ERRORS", "5")

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, align_llr_for_decoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def verify_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 标准蝶形+比特倒序结果（与手算 G_N 一致）
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


def verify_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = align_llr_for_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)


def verify_sc_recursive():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = align_llr_for_decoder(compute_llr(y, sigma))
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2)


def verify_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = align_llr_for_decoder(compute_llr(y, sigma))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def run_all():
    verify_encoder()
    verify_sc_high_snr()
    verify_sc_recursive()
    verify_scl_equals_sc()
    print("verify.py: all checks passed.")


if __name__ == "__main__":
    run_all()

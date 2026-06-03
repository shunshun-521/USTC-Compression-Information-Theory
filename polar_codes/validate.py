"""模块数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import (
    bpsk_modulate,
    awgn_channel,
    compute_llr,
    eb_n0_to_sigma,
    prepare_decoder_llr,
    prepare_frozen_bits_decoder,
    map_decoder_bits_to_natural,
)
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x} != {expected}"
    print("encoder OK")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_dec = prepare_frozen_bits_decoder(frozen_bits, N)

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = prepare_decoder_llr(compute_llr(y, sigma), N)
        u_hat = map_decoder_bits_to_natural(
            sc_decode(llr, frozen_dec), N
        )
        u_hat_r = map_decoder_bits_to_natural(
            sc_decode_recursive(llr, frozen_dec), N
        )
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
        assert np.array_equal(u_hat, u_hat_r), "SC 递归与非递归不一致"
    assert errors == 0, f"SC 无损测试失败: {errors}/100"
    print("SC lossless OK")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_dec = prepare_frozen_bits_decoder(frozen_bits, N)

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = prepare_decoder_llr(compute_llr(y, sigma), N)
        u_sc = map_decoder_bits_to_natural(sc_decode(llr, frozen_dec), N)
        u_scl, _ = SCLDecoder(N, frozen_dec, list_size=1).decode(llr)
        u_scl = map_decoder_bits_to_natural(u_scl, N)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"
    print("SCL L=1 OK")


if __name__ == '__main__':
    test_encoder()
    test_sc_lossless()
    test_scl_equiv_sc()
    print('all validation passed')

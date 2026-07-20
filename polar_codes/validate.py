"""模块正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1]), f"编码器错误"


def validate_sc_decoder():
    N, K = 64, 32
    design_ebn0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_ebn0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)

    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_hat_nr = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat_nr, u_hat_r), "递归与非递归 SC 不一致"
        assert np.array_equal(u_hat_nr[info_idx], payload), "SC 译码错误"


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    rate = K / N
    sigma = eb_n0_to_sigma(5.0, rate)

    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"


def validate_bp_decoder():
    from decoder_bp import BPDecoder

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = 1e-3

    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)

        u_bp, _ = BPDecoder(N, frozen_bits).decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_bp[info_idx], payload), "BP 译码错误"
        assert np.array_equal(u_sc[info_idx], payload), "SC 译码错误"


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)


def run_all_validations():
    validate_encoder()
    validate_sc_decoder()
    validate_scl_equals_sc()
    validate_crc()
    validate_bp_decoder()
    print("所有模块校验通过")


if __name__ == "__main__":
    run_all_validations()

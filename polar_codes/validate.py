"""单元测试：编码器、SC/SCL 一致性、高信噪比无损译码"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction, frozen_mask_from_info
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 G_N = B_N F^{⊗n} 的列向量约定一致：x = (B @ G) @ u (mod 2)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def validate_sc_lossless(num_frames=100):
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits, decode_idx = frozen_mask_from_info(N, info_idx)
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(num_frames):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[decode_idx], payload)


def validate_sc_recursive_match():
    N, K = 128, 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits, decode_idx = frozen_mask_from_info(N, info_idx)
    rng = np.random.default_rng(1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(3.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(a, b)


def validate_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits, decode_idx = frozen_mask_from_info(N, info_idx)
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(4.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def run_all():
    validate_encoder()
    validate_sc_lossless()
    validate_sc_recursive_match()
    validate_scl_l1_equals_sc()
    print("All validations passed.")


if __name__ == "__main__":
    run_all()

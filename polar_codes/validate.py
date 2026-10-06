"""单元测试与模块一致性校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def validate_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x_bf = polar_encode(u)
    x_mx = polar_encode_matrix(u)
    assert np.array_equal(x_bf, x_mx), f"蝶形与矩阵编码不一致: {x_bf} vs {x_mx}"
    # 规格示例 [0,0,1,1] 与 B_N F^{\\otimes n} 矩阵约定不一致，以矩阵校验为准


def validate_sc_lossless(num_frames=100, N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_rec), "递归与非递归 SC 不一致"
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码错误"


def validate_scl_equiv_sc(N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(8.0, rate)
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)


def run_all():
    validate_encoder()
    validate_sc_lossless()
    validate_scl_equiv_sc()
    validate_crc()
    print("All validations passed.")


if __name__ == "__main__":
    run_all()

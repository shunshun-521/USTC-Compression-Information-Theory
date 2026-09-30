"""模块数值校验"""
import numpy as np
from encoder import polar_encode, polar_encode_matrix
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive_channel
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_self_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器与矩阵不一致: {x} vs {x_ref}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info.tolist()) & set(frozen.tolist())) == 0

    assert crc_check(crc_encode(np.array([1, 0, 1, 0, 1, 1]), 8), 8)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != payload):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败，错误帧数={errors}"

    u_hat_sc = sc_decode(llr, frozen_bits)
    u_hat_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_hat_sc, u_hat_scl), "L=1 SCL 应与 SC 一致"

    print("self_test: 全部通过")


if __name__ == "__main__":
    run_self_tests()

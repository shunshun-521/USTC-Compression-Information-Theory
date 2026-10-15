"""模块数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, get_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_all_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = get_generator_matrix(4)
    assert np.array_equal(x, G @ u % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], bits), "SC 译码无损校验失败"

    sigma = eb_n0_to_sigma(15.0, K / N)
    errors = 0
    for _ in range(100):
        bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = bits
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        if not np.array_equal(u_hat[info_idx], bits):
            errors += 1
    assert errors == 0, f"Eb/N0=15dB 下出现 {errors} 次误帧"

    # 递归与非递归一致
    llr = compute_llr(y, sigma)
    assert np.array_equal(
        sc_decode(llr, frozen_bits), sc_decode_recursive(llr, frozen_bits)
    )

    # L=1 SCL 等价 SC
    u_hat_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_hat_scl, sc_decode(llr, frozen_bits))

    # CRC
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)

    print("All validation tests passed.")


if __name__ == "__main__":
    run_all_tests()

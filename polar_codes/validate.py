"""模块数值校验（编码器、SC、SCL L=1）。"""
import numpy as np

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, map_llr_to_decoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validation():
    # 与 u @ B_N @ F^{⊗n} 一致（BPSK 极化码常用行向量约定）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1]), f"编码器错误: {polar_encode(u2)}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = map_llr_to_decoder(compute_llr(y, sigma), N)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], bits)

    # SCL L=1 ~ SC
    llr_test = np.array([1.2, -0.5, 2.0, -1.0, 0.3, -2.1, 0.8, -0.2][:8])
    fb = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    u_sc = sc_decode(llr_test, fb)
    u_scl, _ = SCLDecoder(8, fb, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("validate.py: 所有校验通过")


if __name__ == "__main__":
    run_validation()

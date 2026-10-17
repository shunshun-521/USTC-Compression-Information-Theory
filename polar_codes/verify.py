"""模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all_checks():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], bits):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} 帧错误"

    llr_test = np.array([1.0, -2.0, 0.5, -0.5, 2.0, -1.0, 0.0, 1.5])
    u_rec = sc_decode_recursive(llr_test, np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    u_fast = sc_decode(llr_test, np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    assert np.array_equal(u_rec, u_fast), "非递归 SC 与递归 SC 不一致"

    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = rng.normal(0, 2, N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    print("所有校验通过。")


if __name__ == "__main__":
    run_all_checks()

"""模块数值校验（编码、SC/SCL 一致性、高信噪比无损）。"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validation():
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} 错误帧"

    llr_test = rng.normal(0, 1, N)
    u_rec = sc_decode(llr_test, frozen_bits)
    u_ref = sc_decode_recursive(llr_test, frozen_bits)
    assert np.array_equal(u_rec, u_ref), "非递归与递归 SC 不一致"

    llr_one = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
    u_sc, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_one)
    u_scl = sc_decode(llr_one, frozen_bits)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    print("validate.py: 所有校验通过")


if __name__ == "__main__":
    run_validation()

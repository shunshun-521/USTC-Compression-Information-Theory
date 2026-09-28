"""模块数值校验（各实验脚本开头调用）"""
import os
import numpy as np

from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    if verbose:
        print("Running polar code unit tests...")

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} errors"

    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("All unit tests passed.")


if __name__ == "__main__":
    if os.environ.get("POLAR_SKIP_VALIDATE") == "1":
        print("Skip validate (POLAR_SKIP_VALIDATE=1)")
    else:
        run_unit_tests()

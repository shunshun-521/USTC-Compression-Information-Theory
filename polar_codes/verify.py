"""模块正确性校验（仿真脚本运行前调用）"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode, arikan_generator
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u @ arikan_generator(4)) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info], u_hat[info]):
            errors += 1
    assert errors == 0, f"SC 高 SNR 校验失败: {errors}/100"

    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    if verbose:
        print("verify.py: 所有单元测试通过")


if __name__ == "__main__":
    run_unit_tests()

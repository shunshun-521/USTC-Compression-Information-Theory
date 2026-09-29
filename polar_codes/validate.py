"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validations():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    frozen_bool = frozen.astype(bool)

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bool)
        u_rec = sc_decode_recursive(llr, frozen_bool)
        if not np.array_equal(u_sc, u_rec):
            raise AssertionError("递归与非递归 SC 不一致")
        if not np.array_equal(u_sc[info_idx], u[info_idx]):
            errors += 1
    if errors > 5:
        raise AssertionError(f"SC 高信噪比校验失败: {errors}/100 帧错误")

    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bool = np.ones(N, dtype=bool)
    frozen_bool[info_idx] = False
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(8, K / N))
    u1, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
    u2 = sc_decode(llr, frozen_bool)
    if not np.array_equal(u1, u2):
        raise AssertionError("L=1 SCL 与 SC 不一致")

    print("validate.py: 所有校验通过")


if __name__ == "__main__":
    run_validations()

"""单元测试与快速校验"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from sc_backend import scl_decode_core


def run_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u, apply_bit_reversal=False)
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u, apply_bit_reversal=False)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            err += 1
    assert err <= 2, f"SC 高信噪比测试失败: {err}/100"

    u0 = np.zeros(N, dtype=int)
    u0[info] = rng.integers(0, 2, K)
    llr0 = compute_llr(bpsk_modulate(polar_encode(u0, apply_bit_reversal=False)), sigma)
    uh_sc = sc_decode(llr0, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr0)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"

    print("validate.py: PASS")


if __name__ == "__main__":
    run_all()

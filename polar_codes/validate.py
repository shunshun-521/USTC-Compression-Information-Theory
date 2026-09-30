"""模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sc_fail = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            sc_fail += 1
    assert sc_fail == 0, f"SC 高 SNR 校验失败: {sc_fail}/100"

    u0 = np.zeros(N, dtype=int)
    u0[info] = rng.integers(0, 2, K)
    x0 = polar_encode(u0)
    llr0 = compute_llr(bpsk_modulate(x0), 0.01)
    uh_sc = sc_decode(llr0, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr0)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_validation()

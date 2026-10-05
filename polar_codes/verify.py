"""单元测试与数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    frozen_b = frozen.astype(bool)
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        if not np.array_equal(sc_decode(compute_llr(y, sigma), frozen_b), u):
            sc_err += 1
    assert sc_err == 0, f"SC 无损校验失败: {sc_err}/100"

    u0 = np.zeros(N, dtype=int)
    u0[info] = rng.integers(0, 2, K)
    llr0 = compute_llr(bpsk_modulate(polar_encode(u0)), sigma)
    u_sc = sc_decode(llr0, frozen_b)
    u_scl, _ = SCLDecoder(N, frozen_b, list_size=1).decode(llr0)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    if verbose:
        print("[verify] 编码器 / SC / SCL(L=1) 校验通过")


if __name__ == "__main__":
    run_unit_tests()

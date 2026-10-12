"""模块数值校验"""
import os
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, verify_encoder


def run_unit_tests():
    verify_encoder()

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        ok += int((uh[info] == u[info]).all())
    assert ok >= 95, f"SC 高信噪比测试失败: {ok}/100"

    scl1 = SCLDecoder(N, frozen, list_size=1)
    scl4 = SCLDecoder(N, frozen, list_size=4)
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u1, _ = scl1.decode(llr)
    u4, _ = scl4.decode(llr)
    assert np.array_equal(u1, u4), "L=1 SCL 应与 L=4 在无分裂歧义时一致（高 SNR）"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

"""模块数值校验"""
import os
import numpy as np

os.environ.setdefault("POLAR_QUICK", "1")


def run_unit_tests():
    from encoder import polar_encode
    from construction import ga_construction
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode
    from decoder_scl import SCLDecoder

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"

    info, frozen_idx, _ = ga_construction(8, 4, 2.5)
    print("N=8,K=4 info:", info, "frozen:", frozen_idx)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12, K / N)
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            errs += 1
    assert errs == 0, f"SC 高信噪比校验失败: {errs}/100"

    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), sigma)
    u1 = sc_decode(llr, frozen)
    u2, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u1, u2), "L=1 SCL 应等价于 SC"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()

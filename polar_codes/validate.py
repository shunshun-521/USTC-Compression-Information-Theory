"""数值正确性校验（各实验脚本可 import）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    from encoder import polar_encode_matrix

    assert np.array_equal(x, polar_encode_matrix(u)), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=bool)
    fb[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), fb)
        if np.array_equal(uh[info], u[info]):
            ok += 1
    assert ok == 100, f"SC 高信噪比测试失败: {ok}/100"

    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.2)
    uh_sc = sc_decode(llr, fb)
    uh_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"

    if verbose:
        print("validate.py: 全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

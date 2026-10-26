"""模块数值校验（仿真脚本启动前调用）"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, reorder_llr_for_decoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    br4 = bit_reversal_permutation(4)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(123)

    ok = 0
    for _ in range(100):
        u = np.zeros(N, int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 100 * bpsk_modulate(polar_encode(u))
        llr = reorder_llr_for_decoder(llr, br)
        if np.array_equal(sc_decode(llr, frozen.astype(bool)), u):
            ok += 1
    assert ok == 100, f"SC 无损校验失败: {ok}/100"

    sigma = eb_n0_to_sigma(10.0, 0.5)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = reorder_llr_for_decoder(compute_llr(y, sigma), br)
        if np.array_equal(sc_decode(llr, frozen.astype(bool)), u):
            ok += 1
    assert ok >= 90, f"高信噪比 SC 校验失败: {ok}/100"

    u = np.zeros(N, int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = reorder_llr_for_decoder(100 * bpsk_modulate(polar_encode(u)), br)
    uh_sc = sc_decode(llr, frozen.astype(bool))
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"

    if verbose:
        info8, frozen8, _ = ga_construction(8, 4, 2.5)
        print("N=8 GA info:", info8, "frozen:", frozen8)
        info256, _, _ = ga_construction(256, 128, 2.5)
        print("N=256 info (first 20):", info256[:20])
        print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_unit_tests()

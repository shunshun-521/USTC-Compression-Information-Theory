"""模块正确性校验（各实验脚本开头调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_ok = 0
    for _ in range(100):
        msg = np.zeros(N, dtype=int)
        msg[info] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(msg)), sigma, rng), sigma)
        if np.array_equal(sc_decode(llr, frozen), msg):
            sc_ok += 1
    assert sc_ok == 100, f"SC 无损校验失败: {sc_ok}/100"

    scl = SCLDecoder(N, frozen, list_size=1)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_scl, sc_decode(llr, frozen)), "L=1 SCL 应等价于 SC"

    if verbose:
        info8, frozen8, _ = ga_construction(8, 4, 2.5)
        print("GA N=8:", "info", info8, "frozen", frozen8)
        info256, _, _ = ga_construction(256, 128, 2.5)
        print("GA N=256 info (first 20):", info256[:20])
        print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_validation()

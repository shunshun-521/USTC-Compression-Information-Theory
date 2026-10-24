"""数值正确性校验（各实验脚本开头调用）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    info8, _, _ = ga_construction(8, 4, 2.5)
    frozen8 = np.ones(8, dtype=int)
    frozen8[info8] = 0
    if verbose:
        print("N=8 GA info:", info8)

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, len(info))
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, 0.5))
        if not np.array_equal(sc_decode(llr, frozen), u):
            errs += 1
    assert errs == 0, f"SC 高信噪比校验失败: {errs}/100"

    u0 = np.zeros(N, dtype=int)
    u0[info] = np.random.randint(0, 2, len(info))
    llr0 = compute_llr(bpsk_modulate(polar_encode(u0)), eb_n0_to_sigma(10.0, 0.5))
    uh_sc = sc_decode(llr0, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr0)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"

    if verbose:
        print("validate.py: 全部校验通过。")


if __name__ == "__main__":
    run_validation()

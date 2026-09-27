"""极化码模块单元测试（仿真脚本运行前调用）。"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode
from simulation import run_simulation


def run_unit_tests(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"编码器错误: 得到 {x}, 期望 {expected}")

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info] = False
    rng = np.random.default_rng(0)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info] != u[info]):
            sc_errors += 1
    if sc_errors > 0:
        ok = False
        if verbose:
            print(f"SC 无损校验失败: {sc_errors}/100 帧有误")

    N = 128
    info, _, _ = ga_construction(N, 64, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info] = False
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, 64)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, 0.5))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
        ok = False
        if verbose:
            print("SCL(L=1) 与 SC 不一致")

    if verbose:
        print("单元测试通过" if ok else "单元测试存在失败项")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_unit_tests() else 1)

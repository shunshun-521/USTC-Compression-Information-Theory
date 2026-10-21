"""模块数值校验（仿真前调用）"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    ok = True

    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    if not np.array_equal(x, [0, 0, 1, 1]):
        ok = False
        if verbose:
            print(f"编码器错误: 得到 {x}")

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        if not np.array_equal(sc_decode(llr, frozen), u):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 无损校验失败: {sc_err}/100 帧错误")

    N, K = 128, 64
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    if not np.array_equal(uh_sc, uh_scl):
        ok = False
        if verbose:
            print("SCL L=1 与 SC 不一致")

    if verbose:
        print("单元测试通过" if ok else "单元测试存在失败项")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_unit_tests() else 1)

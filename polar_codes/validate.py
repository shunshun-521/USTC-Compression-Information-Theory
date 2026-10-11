"""模块数值校验（实验脚本启动前调用）"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    ok = True

    # 编码器（与 Vangala SCD 配套的蝶形编码）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"编码器错误: got {x}, expected {expected}")

    # SC 无损
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    sc_ok = True
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(u, sc_decode(llr, frozen)):
            sc_ok = False
            break
    if not sc_ok:
        ok = False
        if verbose:
            print("SC 无损校验失败")

    # L=1 SCL 等价 SC
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = scl.decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            ok = False
            if verbose:
                print("SCL L=1 与 SC 不一致")
            break

    if verbose:
        print("validate:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    run_validation()

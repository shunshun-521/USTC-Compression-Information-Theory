"""模块数值校验（实验前调用）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix_butterfly


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix_butterfly(4)
    if not np.array_equal(x, (u @ G) % 2):
        ok = False
        if verbose:
            print("编码器与 F^{\\otimes n} 不一致", x)

    N, K = 64, 32
    info, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen = np.zeros(N, dtype=bool)
    frozen[frozen_idx] = True
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, K)
        y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        if not np.array_equal(uh, u):
            err += 1
    if err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比测试失败: {err}/100 帧错误")

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=np.int8)
    u[info] = rng.integers(0, 2, K)
    y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
    uh_scl, _ = scl.decode(compute_llr(y, sigma))
    if not np.array_equal(uh_scl, sc_decode(compute_llr(y, sigma), frozen)):
        ok = False
        if verbose:
            print("SCL L=1 与 SC 不一致")

    if verbose:
        print("validate:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    run_validation()

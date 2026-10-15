"""模块数值校验（仿真脚本导入前调用）。"""
import numpy as np

from channel import bpsk_modulate
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    expected = (u @ G) % 2
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"编码器错误: got {x}, expected {expected}")
    elif verbose:
        print("编码器校验通过")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = 1e3 * bpsk_modulate(x)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比校验失败: {sc_err}/100 帧错误")
    elif verbose:
        print("SC 译码校验通过 (N=64, Eb/N0=10dB)")

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    scl_err = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-4)
        u_hat, _ = scl.decode(llr)
        if not np.array_equal(u_hat, u):
            scl_err += 1
    if scl_err > 0:
        ok = False
        if verbose:
            print(f"SCL(L=1) 与 SC 不一致: {scl_err}/20")
    elif verbose:
        print("SCL L=1 路径度量校验通过")

    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_validation() else 1)

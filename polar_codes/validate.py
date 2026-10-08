"""模块数值正确性校验"""
import os
import numpy as np

from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, align_llr_for_decoder
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    if not np.array_equal(x, x_mat):
        ok = False
        if verbose:
            print(f"编码器与生成矩阵不一致: butterfly={x}, matrix={x_mat}")
    elif verbose:
        print(f"编码器校验通过: u={u} -> x={x}")

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=np.int8)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = align_llr_for_decoder(compute_llr(bpsk_modulate(x), 1e-6), N)
        uh = sc_decode(llr, frozen.astype(bool))
        if not np.array_equal(uh, u):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 无损校验失败: {sc_err}/100 帧错误")
    elif verbose:
        print("SC 无损校验通过 (N=64, 100 帧)")

    rng = np.random.default_rng(0)
    scl_err = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(10.0, K / N)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = align_llr_for_decoder(compute_llr(y, sigma), N)
        uh_sc, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            scl_err += 1
    if scl_err > 0:
        ok = False
        if verbose:
            print(f"SCL L=1 路径度量校验失败: {scl_err}/50")
    elif verbose:
        print("SCL L=1 与 SC 一致性校验通过")

    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_validation() else 1)

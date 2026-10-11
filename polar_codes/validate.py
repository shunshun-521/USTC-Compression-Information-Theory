"""模块数值校验（各实验脚本启动时调用）。"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode, generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validations(verbose=True):
    ok = True

    # 编码器：与 G_N = B_N F^{\\otimes n} 一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u @ generator_matrix(4)) % 2
    if not np.array_equal(x, x_ref):
        ok = False
        if verbose:
            print("编码器校验失败:", x, x_ref)

    # GA 构造打印
    if verbose:
        info8, frozen8, _ = ga_construction(8, 4, 2.5)
        print("N=8, K=4, Eb/N0=2.5dB")
        print("info_indices:", info8)
        print("frozen_indices:", frozen8)
        info256, _, _ = ga_construction(256, 128, 2.5)
        print("N=256, K=128, first 20 info:", info256[:20])

    # SC 无损
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sc_ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-8)
        if np.array_equal(sc_decode(llr, frozen_bits), u):
            sc_ok += 1
    if sc_ok < 100:
        ok = False
        if verbose:
            print(f"SC 校验失败: {sc_ok}/100")

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits.astype(bool), list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, len(info_idx))
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-8)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            ok = False
            if verbose:
                print("SCL L=1 与 SC 不一致")
            break

    if verbose:
        print("validate:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    run_validations()

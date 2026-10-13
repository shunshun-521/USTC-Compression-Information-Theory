"""模块数值校验（仿真脚本启动前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"编码器错误: 得到 {x}, 期望 {expected}")

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("N=8, K=4, Eb/N0=2.5dB")
        print("info_indices:", info8)
        print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print("N=256, K=128, first 20 info:", info256[:20])

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)[br]
        if not np.array_equal(sc_decode(llr, frozen), u):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比校验失败: {sc_err}/100 帧错误")

    u_test = np.zeros(N, dtype=int)
    u_test[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_test)), sigma)[br]
    if not np.array_equal(sc_decode(llr, frozen), sc_decode_recursive(llr, frozen)):
        ok = False
        if verbose:
            print("SC 递归与非递归不一致")

    scl = SCLDecoder(N, frozen, list_size=1)
    uh_scl, _ = scl.decode(llr)
    if not np.array_equal(uh_scl, sc_decode(llr, frozen)):
        ok = False
        if verbose:
            print("SCL(L=1) 与 SC 不一致")

    if verbose:
        print("校验通过" if ok else "校验未全部通过")
    return ok


if __name__ == "__main__":
    run_validation()

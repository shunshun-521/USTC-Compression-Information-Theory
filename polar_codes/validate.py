"""模块数值校验（实验脚本启动前调用）"""
import numpy as np

from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode, build_generator_matrix
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    if not np.array_equal(x, x_ref):
        ok = False
        if verbose:
            print("编码器与 G 矩阵不一致", x, x_ref)
    elif verbose:
        print(f"编码器校验通过: u={u} -> x={x}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比校验失败: {sc_err}/100 帧错误")
    elif verbose:
        print("SC 高信噪比校验通过 (N=64, K=32, 100 帧)")

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    scl_err = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh, _ = scl.decode(llr)
        if not np.array_equal(uh, u):
            scl_err += 1
    if scl_err > 0:
        ok = False
        if verbose:
            print(f"SCL(L=1) 与 SC 不一致: {scl_err}/50")
    elif verbose:
        print("SCL L=1 与 SC 等价性校验通过")

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("GA N=8:", "info", info8, "frozen", frozen8)

    if not ok:
        raise AssertionError("validate.py 校验未通过")
    return True


if __name__ == "__main__":
    run_validation()

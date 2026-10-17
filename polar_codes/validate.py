"""模块数值校验（仿真脚本导入运行）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
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

    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = 1e-6
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比校验失败: {sc_err}/100 帧错误")

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    scl_err = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)
        uh, _ = scl.decode(llr)
        uh_sc = sc_decode(llr, frozen_bits)
        if not np.array_equal(uh, uh_sc):
            scl_err += 1
    if scl_err > 0:
        ok = False
        if verbose:
            print(f"SCL(L=1) 与 SC 不一致: {scl_err}/50")

    if verbose and ok:
        print("validate.py: 全部校验通过")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_validation() else 1)

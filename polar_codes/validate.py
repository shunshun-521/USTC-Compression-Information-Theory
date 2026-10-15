"""模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    ok = True

    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"编码器错误: 得到 {x}, 期望 {expected}")

    # GA 构造打印
    info8, frozen8, _ = ga_construction(8, 4, 2.5, rate=0.5)
    info256, _, _ = ga_construction(256, 128, 2.5, rate=0.5)
    if verbose:
        print("N=8, K=4, info:", info8, "frozen:", frozen8)
        print("N=256, K=128, info[:20]:", info256[:20])

    # SC 高信噪比
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比测试失败: {sc_err}/100 帧错误")

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen, list_size=1)
    scl_err = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh, _ = scl.decode(llr)
        if not np.array_equal(uh, u):
            scl_err += 1
    if scl_err > 0:
        ok = False
        if verbose:
            print(f"SCL(L=1) 与 SC 不一致: {scl_err}/20")

    if verbose:
        print("validate:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    run_validation()

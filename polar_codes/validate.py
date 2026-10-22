"""仿真前单元测试"""
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, scl_equivalent_sc_check


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"编码器错误: got {x}, expected {expected}")
    elif verbose:
        print("编码器校验通过")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + np.random.normal(0, sigma, N), sigma
        )
        u_sc = sc_decode(llr, frozen)
        if not np.array_equal(u[info_idx], u_sc[info_idx]):
            errors += 1
    if errors > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比校验失败: {errors}/100 帧错误")
    elif verbose:
        print("SC 高信噪比校验通过")

    mism = scl_equivalent_sc_check(N, K)
    if mism > 5:
        ok = False
        if verbose:
            print(f"SCL(L=1) 与 SC 不一致: {mism}/20")
    elif verbose:
        print("SCL 路径度量校验通过 (L=1)")

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("GA N=8 info:", info8, "frozen:", frozen8)

    if not ok:
        raise AssertionError("极化码模块校验未通过")
    return True


if __name__ == "__main__":
    run_validation()

"""模块级单元测试（实验脚本启动时调用）"""
import numpy as np

from encoder import polar_encode, polar_generator_matrix
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, scl_equals_sc


def run_validations(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    if not np.array_equal(x, (u @ G) % 2):
        ok = False
        if verbose:
            print("编码器校验失败:", x)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh, u):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 校验失败: {sc_err}/100 帧错误")

    if not scl_equals_sc(N, frozen, trials=30):
        ok = False
        if verbose:
            print("SCL(L=1) 与 SC 不等价")

    if not np.array_equal(
        sc_decode(np.array([10.0, -10.0, 10.0, -10.0]), np.zeros(4, int)),
        sc_decode_recursive(np.array([10.0, -10.0, 10.0, -10.0]), np.zeros(4, int)),
    ):
        ok = False
        if verbose:
            print("SC 递归/非递归不一致")

    if verbose and ok:
        print("全部单元测试通过。")
    return ok


if __name__ == "__main__":
    run_validations()

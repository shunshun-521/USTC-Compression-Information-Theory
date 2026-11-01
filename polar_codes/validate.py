#!/usr/bin/env python3
"""模块级单元测试（仿真脚本 import 时执行）"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def _gen_G(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    Fn = F.copy()
    while Fn.shape[0] < N:
        Fn = np.kron(Fn, F)
    n = int(np.log2(N))
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, int(format(i, f"0{n}b")[::-1], 2)] = 1
    return B @ Fn


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器与 G 矩阵不一致: {x}"
    assert np.array_equal(u @ _gen_G(4) % 2, x)

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if np.array_equal(u_hat[info], u[info]):
            ok += 1
    assert ok == 100, f"SC 无损验证失败: {ok}/100"

    u_hat_r = sc_decode_recursive(llr, frozen)
    assert np.array_equal(u_hat_r, u_hat)

    scl = SCLDecoder(N, frozen, list_size=1)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_scl, u_hat), "L=1 的 SCL 应与 SC 一致"


if __name__ == "__main__":
    run_unit_tests()
    print("validate.py: all tests passed")

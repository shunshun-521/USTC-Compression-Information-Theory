"""数值正确性校验脚本。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def build_G(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.block(
            [
                [G, np.zeros((G.shape[0], G.shape[0]), dtype=int)],
                [G, G],
            ]
        )
    n = int(np.log2(N))
    br = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, br[i]] = 1
    return (B @ G) % 2


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_G(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x}, 期望 {x_ref}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u, u_hat)


def test_sc_awgn():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(12.0, K / N)
        y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
        u_hat = sc_decode(compute_llr(y, sigma), frozen)
        err += int(not np.array_equal(u, u_hat))
    assert err == 0, f"SC 高 SNR 测试失败: {err}/100 帧错误"


def test_scl_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(9)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(8.0, K / N)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_sc_awgn()
    test_scl_equals_sc()
    print("verify.py: 全部测试通过")

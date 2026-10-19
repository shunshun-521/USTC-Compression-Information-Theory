"""单元测试与数值校验"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def _build_G(N):
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    n = int(np.log2(N))
    rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])
    return G[rev] % 2


def run_unit_tests():
    N = 4
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = (u @ _build_G(N)) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x} vs {expected}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        if np.array_equal(sc_decode(llr, frozen), u):
            ok += 1
    assert ok == 100, f"SC 无损校验失败: {ok}/100"

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=np.int8)
    u[info] = rng.integers(0, 2, size=K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
    uh, _ = scl.decode(llr)
    assert np.array_equal(uh, sc_decode(llr, frozen)), "L=1 SCL 应等价于 SC"

    info8, _, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", np.where(np.isin(np.arange(8), info8, invert=True))[0])

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])

    return True


if __name__ == "__main__":
    run_unit_tests()
    print("All unit tests passed.")

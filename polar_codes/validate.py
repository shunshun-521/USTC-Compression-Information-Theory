"""共享单元测试与校验函数"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import bit_reversal_permutation, polar_encode


def generate_matrix(n):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def run_unit_tests():
    """数值正确性校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    n = 2
    G = generate_matrix(n)
    br = bit_reversal_permutation(4)
    expected = (u @ G) % 2
    expected = expected[br]
    assert np.array_equal(x, expected), f"编码器错误: {x} vs {expected}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4 -> info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info:", info256[:20])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = 0.05
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    x = polar_encode(u)
    y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc[info_idx], u_scl[info_idx]), "L=1 SCL 应等价于 SC"

    print("全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()

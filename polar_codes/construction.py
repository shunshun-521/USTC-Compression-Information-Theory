"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
    phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    phi(0) = 1
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    zero = x <= 0
    small = (x > 0) & (x < 10)
    large = x >= 10
    y[zero] = 1.0
    xs = x[small]
    y[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[large]
    y[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return y


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 0:
        y = y.reshape(1)
        scalar = True
    else:
        scalar = False
    out = np.zeros_like(y, dtype=np.float64)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    out = (lo + hi) * 0.5
    return float(out[0]) if scalar else out


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回 info_indices, frozen_indices, llr_means
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    sigma = 1.0 / np.sqrt(2.0 * rate) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)
    n = int(np.log2(N))
    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        new_len = len(means) * 2
        m_new = np.zeros(new_len, dtype=np.float64)
        for i, m in enumerate(means):
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi(m)) ** 2)
            m_new[2 * i + 1] = 2.0 * m
        means = m_new
    llr_means = means
    info_indices = np.argsort(-llr_means)[:K]
    info_indices = np.sort(info_indices)
    frozen_indices = np.setdiff1d(np.arange(N), info_indices)
    return info_indices, frozen_indices, llr_means


def build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n} over GF(2)"""
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = np.array([[1]], dtype=np.uint8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    perm = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.uint8)
    for i, p in enumerate(perm):
        B[i, p] = 1
    return (B @ G) % 2


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4:")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info first 20:", info256[:20])

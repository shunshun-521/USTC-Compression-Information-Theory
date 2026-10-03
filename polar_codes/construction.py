"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = (x > 0) & (x < 10)
    large = x >= 10
    tiny = x <= 0
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    xl = x[large]
    out[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    out[tiny] = 1.0
    return out


def phi_inv(y, tol=1e-12, max_iter=80):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 0.0, 1.0 - 1e-15)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        go_left = pm > y
        hi = np.where(go_left, mid, hi)
        lo = np.where(go_left, lo, mid)
        if np.max(hi - lo) < tol:
            break
    return (lo + hi) * 0.5


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    返回 info_indices, frozen_indices, llr_means
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    sigma = 10 ** (-design_eb_n0_db / 20.0) / np.sqrt(2.0 * rate)
    m0 = 2.0 / (sigma ** 2)
    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        ph = phi(m)
        m_new[0::2] = 2.0 * m
        m_new[1::2] = phi_inv(1.0 - (1.0 - ph) ** 2)
        m = m_new
    llr_means = m
    order = np.argsort(-llr_means)
    info_indices = np.sort(order[:K])
    frozen_indices = np.sort(order[K:])
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info first 20:", info256[:20])

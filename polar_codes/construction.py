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
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = (x > 0) & (x < 10)
    large = x >= 10
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    out[large] = (
        np.sqrt(np.pi / x[large])
        * np.exp(-x[large] / 4.0)
        * (1.0 - 10.0 / (7.0 * x[large]))
    )
    # x <= 0 时 phi 未定义；构造中均值恒为正
    out[~(small | large)] = 1.0
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    求 x 使得 phi(x) = y
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, phi(np.array([100.0]))[0], phi(np.array([1e-9]))[0])
    flat = y.ravel()
    lo = np.full(flat.shape, 1e-9)
    hi = np.full(flat.shape, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        hi = np.where(pm > flat, mid, hi)
        lo = np.where(pm <= flat, mid, lo)
    return (lo + hi).reshape(y.shape) / 2.0


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回 info_indices, frozen_indices, llr_means
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)
    n = int(np.log2(N))
    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        new_len = len(m) * 2
        m_new = np.empty(new_len, dtype=np.float64)
        for i, mi in enumerate(m):
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi(mi)) ** 2)
            m_new[2 * i + 1] = 2.0 * mi
        m = m_new
    llr_means = m
    info_indices = np.argsort(llr_means)[-K:]
    info_indices.sort()
    all_idx = np.arange(N)
    frozen_indices = all_idx[~np.isin(all_idx, info_indices)]
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])

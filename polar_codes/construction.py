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
    out = np.zeros_like(x)
    pos = x > 0
    if not np.any(pos):
        return out
    xp = x[pos]
    small = xp < 10
    large = ~small
    if np.any(small):
        xs = xp[small]
        out_flat = out[pos]
        out_flat[small] = np.exp(-0.4527 * xs ** 0.86 + 0.0218)
        out[pos] = out_flat
    if np.any(large):
        xl = xp[large]
        out_flat = out[pos]
        out_flat[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4) * (1 - 10.0 / (7.0 * xl))
        out[pos] = out_flat
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros_like(y)
    flat = y.ravel()
    res = np.zeros_like(flat)
    for idx, yi in enumerate(flat):
        if yi <= 0:
            res[idx] = 0.0
            continue
        if yi >= phi(100.0):
            res[idx] = 100.0
            continue
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if phi(mid) > yi:
                hi = mid
            else:
                lo = mid
        res[idx] = (lo + hi) / 2.0
    return res.reshape(y.shape)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回：
        info_indices, frozen_indices, llr_means
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)
    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        for i, mi in enumerate(m):
            m_new[2 * i] = 2.0 * mi
            m_new[2 * i + 1] = phi_inv(1.0 - (1.0 - phi(mi)) ** 2)
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
    print("N=256, K=128, info_indices (first 20):", info256[:20])

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
    tiny = x <= 0
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    xl = x[large]
    out[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    out[tiny] = 1.0
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    flat = y.ndim == 0
    if flat:
        y = y.reshape(1)
    out = np.zeros_like(y)
    lo = np.full_like(y, 1e-9)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        go_left = pm > y
        lo = np.where(go_left, lo, mid)
        hi = np.where(go_left, mid, hi)
    out = (lo + hi) / 2.0
    return out[0] if flat else out


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
    sigma = 1.0 / np.sqrt(2.0 * rate) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)
    n = int(np.log2(N))
    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        new_len = len(means) * 2
        m_new = np.empty(new_len, dtype=np.float64)
        pm = phi(means)
        m_new[0::2] = phi_inv(1.0 - np.power(1.0 - pm, 2))
        m_new[1::2] = 2.0 * means
        means = m_new
    llr_means = means
    info_indices = np.argsort(llr_means)[-K:]
    info_indices.sort()
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])

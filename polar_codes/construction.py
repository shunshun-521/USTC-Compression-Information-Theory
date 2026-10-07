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
    out[large] = (
        np.sqrt(np.pi / x[large])
        * np.exp(-x[large] / 4.0)
        * (1.0 - 10.0 / (7.0 * x[large]))
    )
    out[tiny] = 1.0
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm > y, mid, lo)
        hi = np.where(pm > y, hi, mid)
    result = (lo + hi) / 2.0
    return result[0] if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    sigma = 10 ** (-design_eb_n0_db / 20.0) / np.sqrt(2.0 * rate)
    m0 = 2.0 / (sigma ** 2)
    n = int(np.log2(N))
    llr_means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_old = llr_means
        m_new = np.empty(2 * len(m_old), dtype=np.float64)
        ph = phi(m_old)
        m_new[0::2] = phi_inv(1.0 - (1.0 - ph) ** 2)
        m_new[1::2] = 2.0 * m_old
        llr_means = m_new
    order = np.argsort(-llr_means)
    info_indices = np.sort(order[:K])
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4:")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])

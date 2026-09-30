"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    small = (x > 0) & (x < 10)
    large = x >= 10
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    xs = x[large]
    out[large] = np.sqrt(np.pi / xs) * np.exp(-xs / 4.0) * (1.0 - 10.0 / (7.0 * xs))
    out[x <= 0] = 1.0
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=float)
    flat = y.ravel()
    res = np.empty_like(flat, dtype=float)
    lo = np.zeros_like(flat)
    hi = np.full_like(flat, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        go_left = pm > flat
        lo = np.where(go_left, lo, mid)
        hi = np.where(go_left, mid, hi)
    res = (lo + hi) / 2.0
    return res.reshape(y.shape)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)
    n = int(np.log2(N))
    llr_means = np.array([m0], dtype=float)
    for _ in range(n):
        m_old = llr_means
        m_new = np.empty(2 * len(m_old), dtype=float)
        phi_m = phi(m_old)
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        m_new[1::2] = 2.0 * m_old
        llr_means = m_new
    info_indices = np.argsort(-llr_means)[:K]
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
    print("N=256 info_indices (first 20):", info256[:20])

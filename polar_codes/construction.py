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
    mask_small = x < 10.0
    xs = x[mask_small]
    out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[~mask_small]
    out[~mask_small] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    result = (lo + hi) * 0.5

    if scalar:
        return float(result[0])
    return result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    返回信息位/冻结位在 u 向量（自然顺序）中的索引。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    sigma = 1.0 / np.sqrt(2.0 * rate) * 10.0 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        phi_m = phi(means)
        m_new = np.empty(len(means) * 2, dtype=np.float64)
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        m_new[1::2] = 2.0 * means
        means = m_new

    llr_means = means
    info_indices = np.sort(np.argsort(llr_means)[-K:])
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
    print("\nN=256, K=128, info_indices (first 20):", info256[:20])

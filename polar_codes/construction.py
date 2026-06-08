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
    mask_small = x < 10
    xs = x[mask_small]
    out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[~mask_small]
    out[~mask_small] = (
        np.sqrt(np.pi / xl)
        * np.exp(-xl / 4.0)
        * (1.0 - 10.0 / (7.0 * xl))
    )
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
    高斯近似构造极化码（矩阵递推，与极化树叶子索引对齐）。

    返回：
        info_indices, frozen_indices, llr_means
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)

    n = int(np.log2(N))
    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = m0

    for j in range(1, n + 1):
        u = 1 << j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bot = z[k + u // 2, j - 1]
                p_top = phi(z_top)
                p_bot = phi(z_bot)
                z[k, j] = phi_inv(1.0 - (1.0 - p_top) * (1.0 - p_bot))
                z[k + u // 2, j] = z_top + z_bot

    llr_means = z[:, n]
    frozen_indices = np.argsort(llr_means, kind="mergesort")[K:]
    frozen_indices = np.sort(frozen_indices)
    info_indices = np.setdiff1d(np.arange(N), frozen_indices)
    info_indices = np.sort(info_indices)

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

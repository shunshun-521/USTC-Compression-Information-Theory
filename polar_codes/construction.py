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
    mask_small = x < 10.0
    xs = x[mask_small]
    out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[~mask_small]
    out[~mask_small] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, phi(np.array([100.0]))[0])
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm > y, mid, lo)
        hi = np.where(pm > y, hi, mid)
    return (lo + hi) * 0.5


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
    sigma = 1.0 / np.sqrt(2.0 * rate * (10.0 ** (design_eb_n0_db / 10.0)))
    m0 = 2.0 / (sigma ** 2)

    n = int(np.log2(N))
    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        half = len(means)
        new = np.empty(2 * half, dtype=np.float64)
        pm = phi(means)
        new[0::2] = phi_inv(1.0 - (1.0 - pm) ** 2)
        new[1::2] = 2.0 * means
        means = new

    llr_means = means
    info_indices = np.argsort(-llr_means)[:K]
    info_indices.sort()
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
    print("N=256, K=128, info_indices (first 20):", info256[:20])

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
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    xl = x[large]
    out[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    # x <= 0: treat as very small positive for numerical stability
    bad = x <= 0
    if np.any(bad):
        out[bad] = phi(np.full(np.sum(bad), 1e-12))
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 0.0, phi(np.array([100.0]))[0])
    lo = np.zeros_like(y)
    hi = np.full_like(y, 1e4)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        # phi(x) 随 x 单调递减
        go_right = pm > y
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
    return (lo + hi) * 0.5


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)
    # GA：Bhattacharyya 参数 z 与均值 m 满足 z = phi(m)
    z = np.array([np.exp(-0.5 / (sigma ** 2))], dtype=np.float64)
    for _ in range(n):
        z_bad = 2.0 * z - z * z
        z_good = z * z
        z = np.concatenate([z_bad, z_good])
    llr_means = -np.log(np.maximum(z, 1e-300))
    order = np.argsort(z)
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
    print("N=256, K=128, info first 20:", info256[:20])

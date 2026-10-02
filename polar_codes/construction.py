"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

from channel import eb_n0_to_sigma


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


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, phi(100.0), phi(1e-12))
    lo = np.full_like(y, 1e-12)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        go_left = pm > y
        lo = np.where(go_left, lo, mid)
        hi = np.where(go_left, mid, hi)
    return (lo + hi) * 0.5


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    返回 info_indices, frozen_indices, llr_means
    """
    if int(np.log2(N)) != np.log2(N):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    sigma = eb_n0_to_sigma(design_eb_n0_db, rate)
    m0 = 2.0 / (sigma ** 2)
    n = int(np.log2(N))
    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        phi_m = phi(m)
        f_branch = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        g_branch = 2.0 * m
        new_m = np.empty(2 * len(m), dtype=np.float64)
        new_m[0::2] = f_branch
        new_m[1::2] = g_branch
        m = new_m
    llr_means = m
    info_indices = np.sort(np.argsort(llr_means)[-K:])
    all_idx = np.arange(N, dtype=int)
    frozen_indices = all_idx[~np.isin(all_idx, info_indices)]
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    for N, K in [(8, 4), (256, 128)]:
        info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
        print(f"N={N}, K={K}")
        print("info_indices:", info_idx)
        print("frozen_indices:", frozen_idx)
        if N == 256:
            print("info_indices (first 20):", info_idx[:20])

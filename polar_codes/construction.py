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
    small = x < 10
    xs = x[small]
    out[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[~small]
    out[~small] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    flat = y.ravel()
    lo = np.zeros_like(flat)
    hi = np.full_like(flat, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm < flat, mid, lo)
        hi = np.where(pm >= flat, mid, hi)
    return mid.reshape(y.shape)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回：
        info_indices: 信息位索引（可靠性最高的 K 个）
        frozen_indices: 冻结位索引
        llr_means: 各合成信道等效 LLR 均值
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
        phi_m = phi(np.minimum(means, 1e2))
        phi_m = np.minimum(phi_m, 1.0 - 1e-8)
        t = 1.0 - (1.0 - phi_m) ** 2
        t = np.clip(t, 1e-10, 1.0 - 1e-10)
        upper = phi_inv(t)
        lower = 2.0 * means
        means = np.concatenate([upper, lower])

    order = np.argsort(means)[::-1]
    info_indices = np.sort(order[:K])
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]
    return info_indices, frozen_indices, means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info first 20:", info256[:20])

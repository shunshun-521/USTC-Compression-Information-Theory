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
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    xl = x[large]
    out[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    # x <= 0: treat as very large LLR (phi -> 0)
    out[x <= 0] = 0.0
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 0.0, phi(np.array([100.0]))[0])
    flat = y.ravel()
    lo = np.zeros_like(flat)
    hi = np.full_like(flat, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        go_left = pm > flat
        lo = np.where(go_left, lo, mid)
        hi = np.where(go_left, mid, hi)
    result = (lo + hi) * 0.5
    return result.reshape(y.shape)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回：
        info_indices: 长度为 K 的信息位索引（升序）
        frozen_indices: 长度为 N-K 的冻结位索引
        llr_means: 长度为 N 的等效 LLR 均值
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    eb_lin = 10.0 ** (design_eb_n0_db / 10.0)
    sigma = 1.0 / np.sqrt(2.0 * rate * eb_lin)
    m0 = 2.0 / (sigma ** 2)

    n = int(np.log2(N))
    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        phi_m = phi(means)
        bad = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        good = 2.0 * means
        means = np.empty(2 * len(means), dtype=np.float64)
        means[0::2] = bad
        means[1::2] = good

    llr_means = means
    info_indices = np.argsort(-llr_means)[:K]
    info_indices = np.sort(info_indices)
    frozen_indices = np.setdiff1d(np.arange(N), info_indices)
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices first 20:", info256[:20])

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
    result = np.empty_like(x)
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.86 + 0.0218)
    xl = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4) * (1 - 10.0 / (7.0 * xl))
    result[x <= 0] = 1.0
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 1e6]）
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 1e6)
    for _ in range(80):
        mid = (lo + hi) / 2
        cmp = phi(mid) < y
        lo = np.where(cmp, mid, lo)
        hi = np.where(cmp, hi, mid)
    result = (lo + hi) / 2
    return float(result[0]) if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    使用递归方式将等效 LLR 均值映射到 Arikan 信道索引。
    """
    if rate is None:
        rate = K / N
    sigma = 1.0 / np.sqrt(2 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / sigma ** 2
    llr_means = np.zeros(N, dtype=np.float64)

    def recurse(length, offset, mean):
        if length == 1:
            llr_means[offset] = mean
            return
        half = length // 2
        bad = phi_inv(1.0 - (1.0 - phi(mean)) ** 2)
        good = 2.0 * mean
        recurse(half, offset, good)
        recurse(half, offset + half, bad)

    recurse(N, 0, m0)
    info_indices = np.sort(np.argsort(llr_means)[-K:])
    frozen_indices = np.setdiff1d(np.arange(N), info_indices)
    return info_indices, frozen_indices, llr_means


if __name__ == '__main__':
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print('N=8, K=4, Eb/N0=2.5dB')
    print('info_indices:', info)
    print('frozen_indices:', frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print('N=256, K=128, info_indices (first 20):', info256[:20])

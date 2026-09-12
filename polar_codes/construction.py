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
    result = np.zeros_like(x)
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.86 + 0.0218)
    xl = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    result = np.zeros_like(y)
    mask_hi = y >= 0.9999
    mask_lo = y <= 1e-10
    mask_mid = ~(mask_hi | mask_lo)

    if np.any(mask_mid):
        lo = np.zeros(np.sum(mask_mid))
        hi = np.full(np.sum(mask_mid), 100.0)
        ym = y[mask_mid]
        for _ in range(60):
            mid = (lo + hi) / 2.0
            pm = phi(mid)
            lo = np.where(pm < ym, mid, lo)
            hi = np.where(pm >= ym, mid, hi)
        result[mask_mid] = (lo + hi) / 2.0

    result[mask_hi] = 100.0
    result[mask_lo] = 0.0
    return float(result[0]) if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10.0 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.zeros(2 * len(m), dtype=np.float64)
        for i in range(len(m)):
            # 1-indexed: m_new[2i-1]=f(bad), m_new[2i]=g(good)
            # 0-indexed: even=f(bad), odd=g(good)
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi(m[i])) ** 2)
            m_new[2 * i + 1] = 2.0 * m[i]
        m = m_new

    llr_means = m
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
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

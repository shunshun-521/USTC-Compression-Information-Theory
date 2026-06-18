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
    result = np.empty_like(x)
    mask_low = x < 10
    mask_high = ~mask_low
    if np.any(mask_low):
        xl = x[mask_low]
        result[mask_low] = np.exp(-0.4527 * np.power(xl, 0.86) + 0.0218)
    if np.any(mask_high):
        xh = x[mask_high]
        result[mask_high] = (
            np.sqrt(np.pi / xh)
            * np.exp(-xh / 4.0)
            * (1.0 - 10.0 / (7.0 * xh))
        )
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        go_right = pm > y
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)

    result = (lo + hi) / 2.0
    return float(result[0]) if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    采用逐层展开 m=1,2,...,N/2 的 GA 递推（与标准实现一致）。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    sigma2 = (1.0 / (2.0 * rate)) * (10 ** (-design_eb_n0_db / 10.0))
    llr0 = 2.0 / sigma2

    llr = np.zeros(N, dtype=np.float64)
    llr[0] = llr0
    m = 1
    while m <= N // 2:
        llr_next = llr.copy()
        for k in range(m):
            temp = llr[k]
            llr_next[2 * k] = phi_inv(1.0 - (1.0 - phi(temp)) ** 2)
            llr_next[2 * k + 1] = 2.0 * temp
        llr = llr_next
        m *= 2

    llr_means = llr
    info_indices = np.sort(np.argsort(llr_means)[-K:])
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

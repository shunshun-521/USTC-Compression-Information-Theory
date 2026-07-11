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
    mask_small = x < 10.0
    mask_large = ~mask_small
    xs = x[mask_small]
    xl = x[mask_large]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    result[mask_large] = (
        np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    )
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm > y, mid, lo)
        hi = np.where(pm <= y, mid, hi)
    result = (lo + hi) / 2.0
    return result.item() if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    sigma2 = (1.0 / (2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 10.0))
    llr0 = 2.0 / sigma2

    llri = np.zeros(N, dtype=np.float64)
    llrcopy = np.zeros(N, dtype=np.float64)
    llri[0] = llr0
    m = 1
    while m <= N // 2:
        for k in range(m):
            llr_temp = llri[k]
            llrcopy[2 * k] = phi_inv(1.0 - (1.0 - phi(llr_temp)) ** 2)
            llrcopy[2 * k + 1] = 2.0 * llr_temp
        llri = llrcopy.copy()
        m *= 2

    llr_means = llri
    sorted_indices = np.argsort(llr_means)
    info_indices = np.sort(sorted_indices[-K:])
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

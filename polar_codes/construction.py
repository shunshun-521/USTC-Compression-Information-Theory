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
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[mask_large]
    result[mask_large] = (
        np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    )
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）。"""
    y = np.asarray(y, dtype=np.float64)
    hi_val = phi(np.array([100.0]))[0]
    y = np.clip(y, 1e-12, hi_val)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    return (lo + hi) / 2.0


def _bit_reversed(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（密度演化，全 N 信道初始化）。

    返回 info_indices, frozen_indices, llr_means。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    eb_no_lin = 10.0 ** (design_eb_n0_db / 10.0) * rate
    z0 = 4.0 * eb_no_lin

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        block = 2 ** j
        half = block // 2
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bot = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bot)))
                z[k + half, j] = z_top + z_bot

    llr_means = z[:, n]
    info_indices = np.sort(np.argsort(-llr_means)[:K])
    frozen_indices = np.sort(np.setdiff1d(np.arange(N), info_indices))
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

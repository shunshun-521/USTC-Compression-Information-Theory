"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似（x > 0）"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_lo = x < 10
    mask_hi = ~mask_lo
    if np.any(mask_lo):
        xl = x[mask_lo]
        out[mask_lo] = np.exp(-0.4527 * np.power(xl, 0.86) + 0.0218)
    if np.any(mask_hi):
        xh = x[mask_hi]
        out[mask_hi] = np.sqrt(np.pi / xh) * np.exp(-xh / 4) * (1 - 10 / (7 * xh))
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        pm = phi(mid)
        go_hi = pm > y
        lo = np.where(go_hi, mid, lo)
        hi = np.where(go_hi, hi, mid)

    result = (lo + hi) / 2
    return result[0] if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（按极化树索引递推，与常用 GA 实现一致）。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    eb_n0_lin = 10 ** (design_eb_n0_db / 10.0)
    z0 = 4.0 * rate * eb_n0_lin

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 2 ** j
        half = u // 2
        for t in range(0, N, u):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                ph_t = phi(z_top)
                ph_b = phi(z_bottom)
                z[k, j] = phi_inv(1.0 - (1.0 - ph_t) * (1.0 - ph_b))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    info_indices = np.sort(np.argsort(llr_means, kind="mergesort")[-K:])
    frozen_indices = np.sort(np.setdiff1d(np.arange(N), info_indices))

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])

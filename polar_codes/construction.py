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
    mask_small = x < 10
    mask_large = ~mask_small
    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.86 + 0.0218)
    xl = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4) * (1 - 10.0 / (7.0 * xl))
    return result


def phi_inv(y, tol=1e-10, max_iter=60):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    is_scalar = y.ndim == 0
    y = np.atleast_1d(y)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = phi(mid)
        lo = np.where(f_mid > y, mid, lo)
        hi = np.where(f_mid > y, hi, mid)
    result = (lo + hi) / 2
    return float(result[0]) if is_scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（密度演化，与极化信道索引对齐）。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    R = K / N if rate is None else rate
    snr_linear = 2.0 * R * (10 ** (design_eb_n0_db / 10.0))
    z0 = 4.0 * snr_linear

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 2 ** j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + u // 2, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + u // 2, j] = z_top + z_bottom

    llr_means = z[:, n]
    frozen_indices = np.sort(np.argsort(llr_means)[: N - K])
    info_indices = np.sort(np.setdiff1d(np.arange(N), frozen_indices))
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x >= 0）
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    mask_zero = x <= 0
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    result[mask_zero] = 1.0
    xs = x[mask_small]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xs = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xs) * np.exp(-xs / 4.0) * (1.0 - 10.0 / (7.0 * xs))
    return np.clip(result, 0.0, 1.0)


def phi_inv(y, tol=1e-10, max_iter=80):
    """phi 函数的数值逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    y = np.clip(y, 0.0, 1.0 - 1e-15)
    result = np.empty_like(y)
    mask_one = y >= 1.0 - 1e-12
    result[mask_one] = 0.0

    y_work = y[~mask_one]
    if y_work.size > 0:
        lo = np.zeros_like(y_work)
        hi = np.full_like(y_work, 1e4)
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            pmid = phi(mid)
            lo = np.where(pmid > y_work, mid, lo)
            hi = np.where(pmid > y_work, hi, mid)
            if np.max(hi - lo) < tol:
                break
        result[~mask_one] = (lo + hi) / 2.0

    return result.item() if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（树形递归，与极化信道分解一致）。

    返回：
        info_indices, frozen_indices, llr_means
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    snr_linear = 2.0 * rate * (10.0 ** (design_eb_n0_db / 10.0))
    m0 = 2.0 * snr_linear

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = m0

    for j in range(1, n + 1):
        block = 1 << j
        half = block // 2
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    info_indices = np.argsort(llr_means)[-K:]
    info_indices = np.sort(info_indices)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.arange(N)[frozen_mask]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

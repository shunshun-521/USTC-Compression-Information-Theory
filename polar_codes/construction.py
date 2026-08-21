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
    mask_small = x < 10.0
    mask_large = ~mask_small

    xs = x[mask_small]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)

    xl = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）。"""
    y = np.asarray(y, dtype=np.float64)
    if np.isscalar(y) or y.ndim == 0:
        return _phi_inv_scalar(float(y))

    flat = y.ravel()
    out = np.array([_phi_inv_scalar(float(v)) for v in flat], dtype=np.float64)
    return out.reshape(y.shape)


def _phi_inv_scalar(y):
    y = float(y)
    if y <= 0.0:
        return 100.0
    if y >= phi(0.0):
        return 0.0

    lo, hi = 0.0, 100.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if phi(mid) > y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（树形密度演化，与信道索引对齐）。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    eb_n0_linear = 10.0 ** (design_eb_n0_db / 10.0)
    z0 = 4.0 * rate * eb_n0_linear

    n = int(np.log2(N))
    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        block = 1 << j
        half = block >> 1
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    sorted_indices = np.argsort(llr_means)[::-1]
    info_indices = np.sort(sorted_indices[:K])
    frozen_indices = np.sort(sorted_indices[K:])
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

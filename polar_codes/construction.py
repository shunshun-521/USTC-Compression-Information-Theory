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
    out = np.empty_like(x)
    mask_small = x < 10.0
    xs = x[mask_small]
    out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[~mask_small]
    out[~mask_small] = (
        np.sqrt(np.pi / xl) * (1.0 - 10.0 / (7.0 * xl)) * np.exp(-xl / 4.0)
    )
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法）。"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    y = np.clip(y, phi(np.array([10000.0]))[0], phi(np.array([1e-6]))[0])

    lo = np.full_like(y, 1e-6)
    hi = np.full_like(y, 10000.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pm = phi(mid)
        lo = np.where(pm > y, lo, mid)
        hi = np.where(pm > y, mid, hi)
    result = 0.5 * (lo + hi)
    return result.item() if scalar else result


def _normalized_design_snr(eb_n0_db, rate):
    """Eb/N0 (dB) -> 归一化设计 SNR（线性）。"""
    return rate * (10 ** (eb_n0_db / 10.0))


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（树形密度演化）。
    """
    if rate is None:
        rate = K / N
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    z0 = 4.0 * _normalized_design_snr(design_eb_n0_db, rate)
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
    frozen_indices = np.argsort(llr_means, kind="mergesort")[K:]
    frozen_indices.sort()
    info_mask = np.ones(N, dtype=bool)
    info_mask[frozen_indices] = False
    info_indices = np.where(info_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info indices:", info256[:20])

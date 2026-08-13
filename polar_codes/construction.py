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
    result[mask_small] = np.exp(-0.4527 * np.power(x[mask_small], 0.86) + 0.0218)
    xs = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xs) * np.exp(-xs / 4) * (1 - 10.0 / (7.0 * xs))
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
        mid = (lo + hi) / 2
        phi_mid = phi(mid)
        lo = np.where(phi_mid > y, mid, lo)
        hi = np.where(phi_mid > y, hi, mid)
    result = (lo + hi) / 2
    return float(result[0]) if scalar else result


def logQ_Borjesson(x):
    """Borjesson 近似的 log Q 函数"""
    a, b = 0.339, 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = np.asarray(x, dtype=float)
    y = -np.log((1 - a) * np.abs(x) + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return np.where(x < 0, np.log(1 - np.exp(y)), y)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（块级密度演化）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    design_snr = 10 ** (design_eb_n0_db / 10) * rate
    z0 = np.full(N, 4.0 * design_snr, dtype=np.float64)

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 1 << j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + u // 2, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + u // 2, j] = z_top + z_bottom

    llr_means = z[:, n]
    reliability = np.array([logQ_Borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)])
    info_indices = np.argsort(reliability, kind='mergesort')[:K]
    info_indices = np.sort(info_indices)
    frozen_indices = np.setdiff1d(np.arange(N), info_indices)

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

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
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    out[mask_small] = np.exp(-0.4527 * np.power(x[mask_small], 0.86) + 0.0218)
    xl = x[mask_large]
    out[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    out[x <= 0] = 1.0
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        hi = np.where(pm > y, mid, hi)
        lo = np.where(pm <= y, mid, lo)
    result = (lo + hi) * 0.5
    return float(result[0]) if scalar else result


def logQ_Borjesson(x):
    """Borjesson 近似 Q 函数对数，用于 GA 信道排序"""
    a, b = 0.339, 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = np.asarray(x, dtype=np.float64)
    x = np.abs(x)
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return np.log(1 - np.exp(y))


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（与标准 GA 密度演化一致）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError(f"N={N} must be a power of 2")

    # 归一化 E_b/N_0（线性）: 4 * R * 10^{Eb/10}
    eb_no_linear = 10 ** (design_eb_n0_db / 10.0) * rate
    z0 = 4.0 * eb_no_linear

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 2**j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + u // 2, j - 1]
                ph_t = phi(z_top)
                ph_b = phi(z_bottom)
                z[k, j] = phi_inv(1.0 - (1.0 - ph_t) * (1.0 - ph_b))
                z[k + u // 2, j] = z_top + z_bottom

    llr_means = z[:, n]
    m_score = np.array(
        [logQ_Borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)]
    )
    frozen_indices = np.argsort(m_score, kind="mergesort")[K:]
    frozen_indices = np.sort(frozen_indices)
    info_indices = np.setdiff1d(np.arange(N), frozen_indices)

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

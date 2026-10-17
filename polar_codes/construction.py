"""
极化码构造：高斯近似（GA）与 Bhattacharyya 界（用于 AWGN 信息位选取）
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似（x > 0）"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    if xs.size:
        out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[mask_large]
    if xl.size:
        out[mask_large] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, phi(100.0), phi(1e-9))
    lo = np.full_like(y, 1e-9)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm > y, lo, mid)
        hi = np.where(pm > y, mid, hi)
    return (lo + hi) * 0.5


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def _ga_llr_means(N, design_eb_n0_db, rate):
    """GA 密度演化得到的各信道 LLR 均值（报告/绘图用）"""
    n = int(np.log2(N))
    eb_no_linear = (10 ** (design_eb_n0_db / 10.0)) * rate
    z0 = 4.0 * eb_no_linear
    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0
    for j in range(1, n + 1):
        u = 2 ** j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + u // 2, j - 1]
                p_top = phi(z_top)
                p_bottom = phi(z_bottom)
                z[k, j] = phi_inv(1.0 - (1.0 - p_top) * (1.0 - p_bottom))
                z[k + u // 2, j] = z_top + z_bottom
    return z[:, n]


def bhattacharyya_construction(N, K, design_eb_n0_db, rate=None):
    """Bhattacharyya 界构造（BPSK-AWGN，与当前编解码器一致）"""
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    eb_no_linear = (10 ** (design_eb_n0_db / 10.0)) * rate
    z0 = -eb_no_linear

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0
    for j in range(1, n + 1):
        u = 2 ** j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + u // 2, j - 1]
                z[k, j] = _logdomain_diff(
                    _logdomain_sum(z_top, z_bottom), z_top + z_bottom
                )
                z[k + u // 2, j] = z_top + z_bottom

    reliabilities = z[:, n]
    frozen_indices = np.argsort(reliabilities, kind="mergesort")[K:]
    frozen_indices = np.sort(frozen_indices)
    info_indices = np.setdiff1d(np.arange(N), frozen_indices)
    info_indices = np.sort(info_indices)
    return info_indices, frozen_indices, reliabilities


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    信息位/冻结位集合采用 Bhattacharyya 界（与 BPSK-AWGN + 当前 Arikan 编解码一致），
    同时返回 GA 密度演化得到的 llr_means 供报告核对。
    """
    if rate is None:
        rate = K / N
    info_indices, frozen_indices, _ = bhattacharyya_construction(
        N, K, design_eb_n0_db, rate=rate
    )
    llr_means = _ga_llr_means(N, design_eb_n0_db, rate)
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4:")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])

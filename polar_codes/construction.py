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
    if xs.size > 0:
        result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[mask_large]
    if xl.size > 0:
        result[mask_large] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = np.array([y])
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm > y, lo, mid)
        hi = np.where(pm > y, mid, hi)
    result = (lo + hi) / 2.0
    return result[0] if scalar else result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def bb_construction(N, K, design_eb_n0_db, rate=None):
    """
    Bhattacharyya 界构造（与 Permuted SC 译码器配套，用于仿真）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    z0 = -rate * (10 ** (design_eb_n0_db / 10.0))
    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        block = 2 ** j
        half = block // 2
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                z[k, j] = _logdomain_diff(
                    _logdomain_sum(z_top, z_bottom), z_top + z_bottom
                )
                z[k + half, j] = z_top + z_bottom

    frozen_indices = np.argsort(z[:, n], kind="mergesort")[K:]
    frozen_indices = np.sort(frozen_indices)
    all_idx = np.arange(N)
    frozen_mask = np.zeros(N, dtype=bool)
    frozen_mask[frozen_indices] = True
    info_indices = all_idx[~frozen_mask]
    llr_means = -z[:, n]
    return info_indices, frozen_indices, llr_means


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（树形密度演化）。

    参数：
        N: 码长（必须是 2 的幂）
        K: 信息位数
        design_eb_n0_db: 设计信噪比 Eb/N0（dB）
        rate: 码率 R=K/N，若为 None 则自动计算

    返回：
        info_indices: 信息位索引（升序）
        frozen_indices: 冻结位索引
        llr_means: 各子信道等效 LLR 均值
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        pm = phi(means)
        m_new = np.empty(2 * len(means), dtype=np.float64)
        m_new[0::2] = phi_inv(1.0 - (1.0 - pm) ** 2)
        m_new[1::2] = 2.0 * means
        means = m_new

    llr_means = means
    info_indices = np.argsort(llr_means)[-K:]
    info_indices = np.sort(info_indices)
    all_idx = np.arange(N)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = all_idx[frozen_mask]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, info_indices (first 20):", info256[:20])

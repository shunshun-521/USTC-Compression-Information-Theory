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
    small = x < 10
    large = ~small

    xs = x[small]
    result[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)

    xl = x[large]
    result[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    if np.isscalar(y) or y.ndim == 0:
        return _phi_inv_scalar(float(y))

    flat = y.ravel()
    out = np.array([_phi_inv_scalar(v) for v in flat], dtype=np.float64)
    return out.reshape(y.shape)


def _phi_inv_scalar(y):
    if y <= 0:
        return 100.0
    if y >= phi(0.0):
        return 0.0

    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if phi(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    参数：
        N: 码长（必须是 2 的幂）
        K: 信息位数
        design_eb_n0_db: 设计信噪比 Eb/N0（dB）
        rate: 码率 R=K/N，若为 None 则自动计算

    返回：
        info_indices: 长度为 K 的数组，信息位在 u 向量中的索引（从 0 开始）
        frozen_indices: 长度为 N-K 的数组，冻结位索引
        llr_means: 长度为 N 的数组，每个极化信道的等效 LLR 均值
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10.0 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)
    n = int(np.log2(N))

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        phi_m = phi(m)
        m_new = np.empty(2 * len(m), dtype=np.float64)
        for i, mi in enumerate(m):
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi_m[i]) ** 2)
            m_new[2 * i + 1] = 2.0 * mi
        m = m_new

    llr_means = m
    info_indices = np.argsort(-llr_means)[:K]
    info_indices.sort()
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

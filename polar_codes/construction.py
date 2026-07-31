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
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.86 + 0.0218)
    xv = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xv) * np.exp(-xv / 4.0) * (1.0 - 10.0 / (7.0 * xv))
    result[x <= 0] = 1.0
    return result


def phi_inv_scalar(y, lo=0.0, hi=100.0, tol=1e-10, max_iter=60):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = float(y)
    if y >= phi(0.0):
        return 0.0
    if y <= phi(hi):
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if phi(mid) > y:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def phi_inv(y):
    """phi 函数的数值逆，支持标量或数组"""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 0:
        return phi_inv_scalar(float(y))
    return np.array([phi_inv_scalar(v) for v in y.flat]).reshape(y.shape)


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
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10.0 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / sigma ** 2

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.zeros(2 * len(m), dtype=np.float64)
        for i in range(len(m)):
            # 1-based: m_new[2i-1]=f (bad), m_new[2i]=g (good)
            # 0-based: m_new[2i]=f, m_new[2i+1]=g
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi(m[i])) ** 2)
            m_new[2 * i + 1] = 2.0 * m[i]
        m = m_new

    llr_means = m
    info_indices = np.sort(np.argsort(llr_means)[-K:])
    all_idx = np.arange(N)
    frozen_indices = all_idx[~np.isin(all_idx, info_indices)]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

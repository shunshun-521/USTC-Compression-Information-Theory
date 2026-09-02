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
    mask_zero = x <= 0

    xs = x[mask_small]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)

    xl = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))

    result[mask_zero] = 1.0
    return result


def _phi_inv_scalar(y):
    """phi 函数的数值逆（二分法，区间 [1e-9, 100]）"""
    y = float(y)
    if y >= 1.0:
        return 0.0
    if y <= 0.0:
        return 100.0

    lo, hi = 1e-9, 100.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if phi(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def phi_inv(y):
    """phi 函数的数值逆，支持标量与数组"""
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 0:
        return _phi_inv_scalar(y_arr.item())

    flat = y_arr.ravel()
    out = np.array([_phi_inv_scalar(val) for val in flat], dtype=np.float64)
    return out.reshape(y_arr.shape)


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
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    if rate is None:
        rate = K / N

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / sigma ** 2

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.zeros(2 * len(m), dtype=np.float64)
        for i, mi in enumerate(m):
            m_new[2 * i] = 2.0 * mi
            pm = phi(mi)
            m_new[2 * i + 1] = phi_inv(1.0 - (1.0 - pm) ** 2)
        m = m_new

    llr_means = m
    sorted_indices = np.argsort(-llr_means)
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

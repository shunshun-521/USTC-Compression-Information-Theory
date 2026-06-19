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
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    if xs.size > 0:
        result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[mask_large]
    if xl.size > 0:
        result[mask_large] = (
            np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
        )
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法）。
    phi(x) 在 x>0 时单调递减，y 很小时 x 很大。
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)
    result = np.empty_like(y)
    phi_hi = phi(np.array([1e-10]))[0]
    for k, yk in enumerate(y.flat):
        if yk <= 0:
            result.flat[k] = 1e10
            continue
        if yk >= phi_hi:
            result.flat[k] = 0.0
            continue
        lo, hi = 0.0, 1e4
        for _ in range(80):
            mid = (lo + hi) / 2.0
            pm = phi(np.array([mid]))[0]
            if pm > yk:
                lo = mid
            else:
                hi = mid
        result.flat[k] = (lo + hi) / 2.0
    return float(result.reshape(y.shape)[0]) if scalar else result.reshape(y.shape)


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

    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)

    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        new_len = len(means) * 2
        m_new = np.empty(new_len, dtype=np.float64)
        for i, m in enumerate(means):
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi(m)) ** 2)
            m_new[2 * i + 1] = 2.0 * m
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
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info indices:", info256[:20])

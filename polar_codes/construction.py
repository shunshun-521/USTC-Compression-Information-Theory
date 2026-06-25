"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x >= 0）
    phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
    phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    phi(0) = 1
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.ones_like(x)
    pos = x > 0
    xp = x[pos]
    mask_small = xp < 10
    mask_large = ~mask_small
    vals = np.empty(xp.shape, dtype=np.float64)
    if np.any(mask_small):
        xs = xp[mask_small]
        vals[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    if np.any(mask_large):
        xl = xp[mask_large]
        vals[mask_large] = np.maximum(
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl)),
            1e-300,
        )
    result[pos] = vals
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，phi 关于 x 单调递减）
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    results = np.empty(y.shape, dtype=np.float64)
    for idx, yi in enumerate(y.flat):
        if yi >= 1.0:
            results.flat[idx] = 0.0
            continue
        if yi <= 0:
            results.flat[idx] = 1e6
            continue
        lo, hi = 0.0, 10.0
        while phi(hi) >= yi:
            hi *= 2.0
            if hi > 1e6:
                break
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if phi(mid) >= yi:
                lo = mid
            else:
                hi = mid
        results.flat[idx] = (lo + hi) / 2.0

    return float(results[0]) if scalar else results.reshape(y.shape)


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

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
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
    print("N=256, K=128, info_indices (first 20):", info256[:20])

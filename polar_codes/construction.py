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
    result = np.zeros_like(x)
    mask_low = (x > 0) & (x < 10)
    mask_high = x >= 10
  # x <= 0 保持为 0
    if np.any(mask_low):
        xl = x[mask_low]
        result[mask_low] = np.exp(-0.4527 * xl ** 0.86 + 0.0218)
    if np.any(mask_high):
        xh = x[mask_high]
        result[mask_high] = np.sqrt(np.pi / xh) * np.exp(-xh / 4) * (1 - 10.0 / (7.0 * xh))
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    result = np.zeros_like(y)
    flat_y = y.ravel()
    flat_result = np.zeros_like(flat_y)

    for idx, yi in enumerate(flat_y):
        if yi <= 0:
            flat_result[idx] = 0.0
            continue
        if yi >= phi(100.0):
            flat_result[idx] = 100.0
            continue
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if phi(mid) < yi:
                lo = mid
            else:
                hi = mid
        flat_result[idx] = (lo + hi) / 2.0

    result[:] = flat_result.reshape(y.shape)
    return result


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
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    # sigma = 1 / sqrt(2*R) * 10^{-Eb/N0/20}
    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / sigma ** 2

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.zeros(2 * len(m), dtype=np.float64)
        phi_m = phi(m)
        # f 分支（坏方向）：索引 2i（0-based 偶数位置）
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        # g 分支（好方向）：索引 2i+1（0-based 奇数位置）
        m_new[1::2] = 2.0 * m
        m = m_new

    llr_means = m
    # 取最大的 K 个 llr_means 对应索引作为 info_indices
    info_indices = np.argsort(llr_means)[-K:]
    info_indices = np.sort(info_indices)
    frozen_indices = np.setdiff1d(np.arange(N), info_indices)

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info_idx, frozen_idx, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info_idx)
    print("frozen_indices:", frozen_idx)

    info_idx256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info_indices:", info_idx256[:20])

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
    mask_lo = x < 10
    mask_hi = ~mask_lo
    xl = x[mask_lo]
    xh = x[mask_hi]
    out[mask_lo] = np.exp(-0.4527 * np.power(xl, 0.86) + 0.0218)
    out[mask_hi] = np.sqrt(np.pi / xh) * np.exp(-xh / 4.0) * (1.0 - 10.0 / (7.0 * xh))
    # x <= 0 的边界（理论上不应出现）
    out[x <= 0] = 1.0
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, phi(np.array([100.0]))[0])
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        go_hi = pm > y
        lo = np.where(go_hi, mid, lo)
        hi = np.where(go_hi, hi, mid)
    return (lo + hi) / 2.0


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)

    n = int(np.log2(N))
    llr_means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(llr_means), dtype=np.float64)
        p_m = phi(llr_means)
        m_new[0::2] = phi_inv(1.0 - (1.0 - p_m) ** 2)
        m_new[1::2] = 2.0 * llr_means
        llr_means = m_new

    # GA 文献中索引常从 1 开始；此处使用 0..N-1，取 LLR 均值最大的 K 个
    info_indices = np.argsort(-llr_means)[:K]
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

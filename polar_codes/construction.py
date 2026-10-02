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
    out = np.empty_like(x)
    small = x < 10.0
    large = ~small
    xs = x[small]
    if xs.size:
        out[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[large]
    if xl.size:
        out[large] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    求 x 使得 phi(x) = y
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, phi(np.array([100.0]))[0])
    flat = y.ravel()
    lo = np.zeros_like(flat)
    hi = np.full_like(flat, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        go_left = pm > flat
        lo = np.where(go_left, lo, mid)
        hi = np.where(go_left, mid, hi)
    result = (lo + hi) * 0.5
    return result.reshape(y.shape)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    sigma = eb_n0_to_sigma_from_design(design_eb_n0_db, rate)
    m0 = 2.0 / (sigma * sigma)
    n = int(np.log2(N))
    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(means), dtype=np.float64)
        ph = phi(means)
        m_new[0::2] = phi_inv(1.0 - (1.0 - ph) ** 2)
        m_new[1::2] = 2.0 * means
        means = m_new
    llr_means = means
    info_indices = np.argsort(-llr_means)[:K]
    info_indices = np.sort(info_indices)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]
    return info_indices, frozen_indices, llr_means


def eb_n0_to_sigma_from_design(design_eb_n0_db, rate):
    from channel import eb_n0_to_sigma as _sigma

    return _sigma(design_eb_n0_db, rate)


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4:")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, means256 = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info first 20:", info256[:20])

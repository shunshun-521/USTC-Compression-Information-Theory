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
    mask_lo = x < 10.0
    mask_hi = ~mask_lo
    xl = x[mask_lo]
    if xl.size:
        out[mask_lo] = np.exp(-0.4527 * np.power(xl, 0.86) + 0.0218)
    xh = x[mask_hi]
    if xh.size:
        out[mask_hi] = (
            np.sqrt(np.pi / xh)
            * np.exp(-xh / 4.0)
            * (1.0 - 10.0 / (7.0 * xh))
        )
    # x <= 0 时 GA 中通常不会出现；数值保护
    out = np.where(x <= 0, 1.0, out)
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, phi(np.array([100.0]))[0])

    def _phi_inv_scalar(target):
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if phi(np.array([mid]))[0] > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    flat = y.ravel()
    inv = np.array([_phi_inv_scalar(t) for t in flat], dtype=np.float64)
    return inv.reshape(y.shape)


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
        for i, m in enumerate(llr_means):
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi(np.array([m]))[0]) ** 2)
            m_new[2 * i + 1] = 2.0 * m
        llr_means = m_new

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

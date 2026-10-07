"""
极化码构造：高斯近似（GA）方法
"""
import numpy as np
from encoder import bit_reversal_permutation


def phi(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = (x > 0) & (x < 10)
    large = x >= 10
    tiny = x <= 0
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    out[large] = (
        np.sqrt(np.pi / x[large])
        * np.exp(-x[large] / 4.0)
        * (1.0 - 10.0 / (7.0 * x[large]))
    )
    out[tiny] = 1.0
    return out


def phi_inv(y):
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)
    y = np.clip(y, 1e-12, 0.999999)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        hi = np.where(pm > y, mid, hi)
        lo = np.where(pm > y, lo, mid)
    result = (lo + hi) * 0.5
    return result[0] if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)
    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(len(means) * 2, dtype=np.float64)
        for i, mi in enumerate(means):
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi(mi)) ** 2)
            m_new[2 * i + 1] = 2.0 * mi
        means = m_new

    llr_means = means
    br = bit_reversal_permutation(N)
    channel_means = np.array([llr_means[br[i]] for i in range(N)], dtype=np.float64)
    info_indices = np.sort(np.argsort(-channel_means)[:K])
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
    print("N=256, K=128, first 20 info:", info256[:20])

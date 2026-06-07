"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

from encoder import bit_reversal_permutation


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
    phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    mask_small = x < 10
    xs = x[mask_small]
    xl = x[~mask_small]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    if xl.size:
        result[~mask_small] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    return result


def _phi_inv_scalar(y):
    lo, hi = 1e-6, 100.0
    y = float(y)
    phi_lo = float(phi(np.array([lo]))[0])
    phi_hi = float(phi(np.array([hi]))[0])
    y = max(min(y, phi_lo), phi_hi)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if float(phi(np.array([mid]))[0]) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [1e-6, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    return np.vectorize(_phi_inv_scalar)(y)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        phi_m = phi(means)
        m_new = np.zeros(len(means) * 2, dtype=np.float64)
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        m_new[1::2] = 2.0 * means
        means = m_new

    llr_means = means
    br = bit_reversal_permutation(N)
    reliabilities = llr_means[br]
    info_indices = np.sort(np.argsort(reliabilities)[-K:])
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

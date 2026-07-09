"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np
from encoder import bit_reversal_permutation

MAX_LLR_MEAN = 15.0


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
  phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
  phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    mask_pos_small = (x > 0) & (x < 10)
    mask_pos_large = x >= 10

    result[mask_pos_small] = np.exp(
        -0.4527 * np.power(x[mask_pos_small], 0.86) + 0.0218
    )
    xs = x[mask_pos_large]
    result[mask_pos_large] = (
        np.sqrt(np.pi / xs)
        * np.exp(-xs / 4.0)
        * (1.0 - 10.0 / (7.0 * xs))
    )
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    result = np.zeros_like(y)
    for idx, yi in enumerate(y):
        if yi <= 1e-12:
            result[idx] = 0.0
            continue
        if yi >= 1.0 - 1e-12:
            result[idx] = MAX_LLR_MEAN
            continue
        lo, hi = 0.0, MAX_LLR_MEAN
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if phi(mid) < yi:
                lo = mid
            else:
                hi = mid
        result[idx] = (lo + hi) / 2.0

    return float(result[0]) if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    assert 2**n == N, "N must be a power of 2"

    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = min(2.0 / (sigma ** 2), MAX_LLR_MEAN)

    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        new_means = np.empty(2 * len(means), dtype=np.float64)
        for i, m in enumerate(means):
            pm = phi(m)
            if pm >= 1.0 - 1e-10:
                bad = 0.0
            else:
                bad = phi_inv(1.0 - (1.0 - pm) ** 2)
            new_means[2 * i] = bad
            new_means[2 * i + 1] = min(2.0 * m, MAX_LLR_MEAN)
        means = new_means

    br = bit_reversal_permutation(N)
    llr_means = means[br]
    sorted_indices = np.argsort(llr_means)
    info_indices = np.sort(sorted_indices[-K:])
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

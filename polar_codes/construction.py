"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

from encoder import bit_reversal_permutation


def phi(x):
    """GA 中的 phi 函数近似（x > 0）"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    xl = x[mask_large]
    out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    out[mask_large] = (
        np.sqrt(np.pi / xl)
        * np.exp(-xl / 4.0)
        * (1.0 - 10.0 / (7.0 * xl))
    )
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 0:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        cmp = phi(mid) > y
        lo = np.where(cmp, mid, lo)
        hi = np.where(cmp, hi, mid)
    result = (lo + hi) / 2.0
    return float(result[0]) if result.size == 1 else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    返回自然顺序下的 info / frozen 索引。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        pm = phi(m)
        m_new[0::2] = phi_inv(1.0 - (1.0 - pm) ** 2)
        m_new[1::2] = 2.0 * m
        m = m_new

    br = bit_reversal_permutation(N)
    llr_natural = np.empty(N, dtype=np.float64)
    llr_natural[br] = m

    info_indices = np.argsort(llr_natural)[-K:]
    info_indices = np.sort(info_indices)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.arange(N)[frozen_mask]

    return info_indices, frozen_indices, llr_natural


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

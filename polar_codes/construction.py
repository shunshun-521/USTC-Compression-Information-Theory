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
    result = np.empty_like(x)
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    xl = x[mask_large]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    result[mask_large] = (
        np.sqrt(np.pi / xl)
        * np.exp(-xl / 4.0)
        * (1.0 - 10.0 / (7.0 * xl))
    )
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        phi_mid = phi(mid)
        lo = np.where(phi_mid > y, mid, lo)
        hi = np.where(phi_mid > y, hi, mid)

    result = (lo + hi) / 2.0
    return result.item() if scalar else result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(np.binary_repr(i, width=n)[::-1], 2)
    return rev


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    信息位索引在比特倒序域中选取（与标准极化码实现一致）。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError(f"N={N} must be a power of 2")

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        phi_m = phi(m)
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        m_new[1::2] = 2.0 * m
        m = m_new

    llr_means = m
    rev_idx = bit_reversal_permutation(N)
    rel_at_rev = llr_means[rev_idx]
    sorted_rev = rev_idx[np.argsort(rel_at_rev)]
    info_indices = np.sort(sorted_rev[-K:])
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

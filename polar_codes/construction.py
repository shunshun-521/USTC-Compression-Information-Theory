"""
极化码构造：高斯近似（GA）方法 + 3GPP 5G 可靠性序列
适用于 BPSK-AWGN 信道，编码矩阵 G_N = F^{\\otimes n}
"""
import os

import numpy as np

_POLAR_5G_CSV = os.path.join(os.path.dirname(__file__), "polar_5G.csv")
_POLAR_5G_TABLE = None


def _load_5g_table():
    global _POLAR_5G_TABLE
    if _POLAR_5G_TABLE is None:
        _POLAR_5G_TABLE = np.genfromtxt(_POLAR_5G_CSV, delimiter=";").astype(int)
    return _POLAR_5G_TABLE


def phi(x):
    """GA 中的 phi 函数近似"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x < 10.0
    out[mask_small] = np.exp(-0.4527 * np.power(x[mask_small], 0.86) + 0.0218)
    xs = x[~mask_small]
    out[~mask_small] = np.sqrt(np.pi / xs) * np.exp(-xs / 4.0) * (1.0 - 10.0 / (7.0 * xs))
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        cmp = phi(mid) >= y
        lo = np.where(cmp, lo, mid)
        hi = np.where(cmp, mid, hi)
    result = (lo + hi) / 2.0
    return result[0] if scalar else result


def _ga_llr_means(N, design_eb_n0_db, rate):
    """计算 GA 等效 LLR 均值（用于报告/核对）"""
    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)
    m = np.array([m0], dtype=np.float64)
    n = int(np.log2(N))
    for _ in range(n):
        phi_m = phi(m)
        m_new = np.empty(2 * len(m), dtype=np.float64)
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        m_new[1::2] = 2.0 * m
        m = m_new
    return m


def _5g_info_indices(N, K):
    """3GPP TS 38.212 可靠性序列（与 G=F^{⊗n} 编码一致）"""
    ch_order = _load_5g_table()
    ind = np.argsort(ch_order[:, 1])
    ch_order_sort = ch_order[ind]
    ch_n = ch_order_sort[:N]
    ind_n = np.argsort(ch_n[:, 0])
    ch_sorted = ch_n[ind_n]
    frozen = ch_sorted[: N - K, 1].astype(int)
    info = ch_sorted[N - K :, 1].astype(int)
    return np.sort(info), np.sort(frozen)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    极化码构造：信息位采用 3GPP 5G 可靠性序列；返回 GA LLR 均值供核对。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    llr_means = _ga_llr_means(N, design_eb_n0_db, rate)
    info_indices, frozen_indices = _5g_info_indices(N, K)
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, info_indices (first 20):", info256[:20])

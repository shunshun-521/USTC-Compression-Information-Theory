"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import os

import numpy as np

# 3GPP 极化码子信道可靠性排序（与 Sionna polar_5G.csv 一致）
_POLAR_5G_CSV = None


def _load_5g_channel_order():
    global _POLAR_5G_CSV
    if _POLAR_5G_CSV is None:
        csv_path = os.path.join(os.path.dirname(__file__), "polar_5G.csv")
        if not os.path.isfile(csv_path):
            try:
                from importlib_resources import files, as_file
                from pathlib import Path

                source = files("sionna.phy.fec.polar.codes").joinpath("polar_5G.csv")
                with as_file(source) as src:
                    data = np.genfromtxt(src, delimiter=";")
            except Exception:
                raise FileNotFoundError(
                    "polar_5G.csv not found; copy from Sionna or ship with polar_codes/"
                )
        else:
            data = np.genfromtxt(csv_path, delimiter=";")
        _POLAR_5G_CSV = data.astype(int)
    return _POLAR_5G_CSV


def select_info_indices(N, K):
    """
    按 3GPP 可靠性排序选取 K 个信息位索引（与当前编码/SC 译码器匹配）。
    """
    ch_order = _load_5g_channel_order()
    ind = np.argsort(ch_order[:, 0])
    ch_order_sort = ch_order[ind]
    ch_order_sort_n = []
    for row in ch_order_sort:
        if int(row[1]) < N:
            ch_order_sort_n.append(row)
        if len(ch_order_sort_n) == N:
            break
    if len(ch_order_sort_n) < N:
        raise ValueError(f"Cannot build length-{N} polar code from ranking table")
    ch_order_sort_n = np.array(ch_order_sort_n)
    ind_n = np.argsort(ch_order_sort_n[:, 0])
    ch_order_n = ch_order_sort_n[ind_n]
    info = np.sort(ch_order_n[N - K :, 1].astype(int))
    return info


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    xl = x[mask_large]
    if xs.size:
        out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    if xl.size:
        out[mask_large] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = phi(mid)
        hi = np.where(val > y, mid, hi)
        lo = np.where(val <= y, mid, lo)
    result = (lo + hi) / 2.0
    return float(result[0]) if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回：
        info_indices, frozen_indices, llr_means
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2**n == N, "N must be a power of 2"

    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma**2)

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        phi_m = phi(m)
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        m_new[1::2] = 2.0 * m
        m = m_new

    llr_means = m
    info_indices = np.sort(np.argsort(llr_means)[-K:])
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]

    return info_indices, frozen_indices, llr_means


def ga_construction_for_simulation(N, K, design_eb_n0_db, rate=None):
    """
    仿真用信息位选取：GA 用于报告；实际仿真采用 3GPP 可靠性排序以匹配 SC 译码器。
    返回 info_indices, frozen_indices（与 ga_construction 相同接口）。
    """
    info_indices = select_info_indices(N, K)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]
    _, _, llr_means = ga_construction(N, K, design_eb_n0_db, rate)
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, info_indices (first 20):", info256[:20])

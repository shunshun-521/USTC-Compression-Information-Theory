"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x < 10.0
    mask_large = ~mask_small
    xs = x[mask_small]
    xl = x[mask_large]
    out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    out[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return out


def phi_inv(y):
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    result = (lo + hi) / 2.0
    return result[0] if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    llr_means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(llr_means), dtype=np.float64)
        phi_m = phi(llr_means)
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        m_new[1::2] = 2.0 * llr_means
        llr_means = m_new

    sorted_indices = np.argsort(llr_means)
    frozen_indices = sorted_indices[: N - K].copy()
    if 0 not in frozen_indices:
        frozen_indices[-1] = 0
        frozen_indices = np.sort(frozen_indices)

    frozen_set = set(frozen_indices.tolist())
    info_indices = np.array(sorted(set(range(N)) - frozen_set), dtype=int)

    if len(info_indices) != K:
        frozen_indices = sorted_indices[: N - K]
        if 0 not in frozen_indices:
            frozen_indices = np.sort(np.append(frozen_indices[:-1], 0))
        info_indices = np.array(sorted(set(range(N)) - set(frozen_indices.tolist())), dtype=int)

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info:", info256[:20])

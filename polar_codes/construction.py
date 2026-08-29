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
    mask_small = x < 10
    mask_large = ~mask_small

    xs = x[mask_small]
    if xs.size > 0:
        out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)

    xl = x[mask_large]
    if xl.size > 0:
        out[mask_large] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    return out


def phi_inv(y, tol=1e-12, max_iter=60):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = phi(mid)
        go_left = f_mid > y
        lo = np.where(go_left, lo, mid)
        hi = np.where(go_left, mid, hi)
        if np.max(hi - lo) < tol:
            break

    result = (lo + hi) / 2.0
    return result.item() if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    返回信息位索引（可靠性最高的 K 个信道）。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    eb_no_linear = 10.0 ** (design_eb_n0_db / 10.0)
    design_snr_norm = eb_no_linear * rate
    m0 = 4.0 * design_snr_norm

    n = int(np.log2(N))
    means = np.array([m0], dtype=np.float64)

    for _ in range(n):
        half = len(means)
        new_means = np.empty(2 * half, dtype=np.float64)
        p = phi(means)
        new_means[0::2] = phi_inv(1.0 - (1.0 - p) ** 2)
        new_means[1::2] = 2.0 * means
        means = new_means

    llr_means = means
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
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

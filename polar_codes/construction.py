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
    result = np.empty_like(x)
    mask_small = x < 10
    mask_large = ~mask_small
    result[mask_small] = np.exp(-0.4527 * np.power(x[mask_small], 0.86) + 0.0218)
    xs = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xs) * np.exp(-xs / 4.0) * (1.0 - 10.0 / (7.0 * xs))
    return result


def phi_inv(y, tol=1e-12, max_iter=100):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 10000.0)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = phi(mid)
        # phi(x) 随 x 单调递减
        go_left = f_mid > y
        lo = np.where(go_left, mid, lo)
        hi = np.where(go_left, hi, mid)
        if np.all(hi - lo < tol):
            break
    result = (lo + hi) / 2.0
    return result.item() if scalar else result


def logQ_Borjesson(x):
    """将等效 LLR 均值映射为可靠性度量（越小越可靠）"""
    a, b = 0.339, 5.510
    half_log2pi = 0.5 * np.log(2.0 * np.pi)
    x = np.abs(x)
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2.0) - half_log2pi
    return np.log(1.0 - np.exp(y))


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（基于极化信道树的 GA 密度演化）。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10.0 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = m0

    for j in range(1, n + 1):
        block = 1 << j
        half = block // 2
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                phi_top = phi(z_top)
                phi_bottom = phi(z_bottom)
                z[k, j] = phi_inv(1.0 - (1.0 - phi_top) * (1.0 - phi_bottom))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    reliabilities = np.array([logQ_Borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)])
    info_indices = np.argsort(reliabilities, kind='mergesort')[-K:]
    info_indices = np.sort(info_indices)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.arange(N)[frozen_mask]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4:")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info indices:", info256[:20])

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
    mask_small = x <= 10
    mask_large = ~mask_small
    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.859 + 0.0218)
    xs = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xs) * np.exp(-xs / 4) * (1 - 10.0 / (7.0 * xs))
    return result


def _phi_derivative(x):
    if x <= 10:
        px = phi(x)
        return -0.4527 * 0.86 * (x ** (-0.14)) * px
    return np.sqrt(np.pi) * np.exp(-x / 4) * (
        (15.0 / 7.0) * (x ** (-2.5)) - (1.0 / 7.0) * (x ** (-1.5)) - 0.25 * (x ** (-0.5))
    )


def phi_inv(y):
    """phi 函数的数值逆（牛顿迭代法）。"""
    y = float(y)
    if 0.0388 <= y <= 1.0221:
        return ((0.0218 - np.log(y)) / 0.4527) ** (1.0 / 0.86)
    x0 = 0.0388
    x1 = x0 - (phi(x0) - y) / _phi_derivative(x0)
    gap = 1e-3
    while abs(x1 - x0) >= gap:
        x0 = x1
        x1 = x1 - (phi(x1) - y) / _phi_derivative(x1)
        if x1 > 1e2:
            gap = 10.0
    return x1


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """高斯近似构造极化码。"""
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N

    sigma2 = (1.0 / (2.0 * rate)) * (10 ** (-design_eb_n0_db / 10.0))
    m0 = 2.0 / sigma2

    llr = np.zeros(N, dtype=np.float64)
    llr[0] = m0
    block = 1
    while block <= N // 2:
        llr_copy = llr.copy()
        for k in range(block):
            temp = llr[k]
            llr_copy[2 * k] = phi_inv(1.0 - (1.0 - phi(temp)) ** 2)
            llr_copy[2 * k + 1] = 2.0 * temp
        llr = llr_copy
        block *= 2

    llr_means = llr
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
    print("N=256, K=128, first 20 info_indices:", info256[:20])

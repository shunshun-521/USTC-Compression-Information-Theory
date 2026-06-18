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
    xl = x[mask_large]
    out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    out[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    phi 关于 x 单调递减
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)
    out = np.empty_like(y)
    for idx, target in enumerate(y.flat):
        lo, hi = 0.0, 100.0
        if target <= 0:
            out.flat[idx] = hi
            continue
        if target >= phi(lo):
            out.flat[idx] = lo
            continue
        for _ in range(60):
            mid = (lo + hi) / 2.0
            pm = phi(mid)
            if pm > target:
                lo = mid
            else:
                hi = mid
        out.flat[idx] = (lo + hi) / 2.0
    return out.item() if scalar else out


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回：
        info_indices: 信息位索引（升序）
        frozen_indices: 冻结位索引（升序）
        llr_means: 各合成信道等效 LLR 均值
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    snr_linear = 2.0 * rate * (10 ** (design_eb_n0_db / 10.0))
    z0 = 2.0 / (1.0 / snr_linear)  # 等价于 2/sigma^2，与初始 LLR 均值一致

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        block = 2 ** j
        half = block // 2
        for start in range(0, N, block):
            for s in range(half):
                k = start + s
                top = z[k, j - 1]
                bottom = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(top)) * (1.0 - phi(bottom)))
                z[k + half, j] = top + bottom

    llr_means = z[:, n]
    frozen_indices = np.argsort(llr_means, kind="mergesort")[: N - K]
    frozen_indices = np.sort(frozen_indices)
    info_indices = np.setdiff1d(np.arange(N), frozen_indices)

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

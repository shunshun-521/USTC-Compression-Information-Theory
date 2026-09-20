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
    mask_small = x < 10.0
    mask_large = ~mask_small

    xs = x[mask_small]
    if xs.size > 0:
        result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)

    xl = x[mask_large]
    if xl.size > 0:
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
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)

    result = (lo + hi) * 0.5
    return result.item() if scalar else result


def _log_q_borjesson(x):
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2.0 * np.pi)
    x = np.abs(x)
    y = -np.log((1.0 - a) * x + a * np.sqrt(b + x * x)) - (x * x) / 2.0 - half_log2pi
    return np.log(1.0 - np.exp(y))


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    采用分层 GA 密度演化，并用 Borjesson 近似将均值 LLR 映射为
    等效 Bhattacharyya 参数以排序选信道。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    eb_no_linear = 10 ** (design_eb_n0_db / 10.0) * rate
    z = np.full(N, 4.0 * eb_no_linear, dtype=np.float64)

    for j in range(1, n + 1):
        block = 1 << j
        z_new = z.copy()
        half = block // 2
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k]
                z_bottom = z[k + half]
                z_new[k] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z_new[k + half] = z_top + z_bottom
        z = z_new

    llr_means = z
    reliabilities = np.array(
        [_log_q_borjesson(0.707 * np.sqrt(max(val, 1e-30))) for val in llr_means]
    )
    frozen_indices = np.argsort(reliabilities)[K:]
    frozen_indices.sort()
    info_indices = np.setdiff1d(np.arange(N), frozen_indices, assume_unique=True)

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

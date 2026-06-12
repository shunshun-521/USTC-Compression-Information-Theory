"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0），向量化实现。
    phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
    phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    if scalar:
        x = x.reshape(1)

    out = np.empty_like(x)
    mask = x < 10.0
    out[mask] = np.exp(-0.4527 * (x[mask] ** 0.86) + 0.0218)
    xm = x[~mask]
    out[~mask] = np.sqrt(np.pi / xm) * np.exp(-xm / 4.0) * (1.0 - 10.0 / (7.0 * xm))

    return out.item() if scalar else out.reshape(x.shape)


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）。"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    out = np.empty_like(y)
    for idx, yi in enumerate(y.flat):
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if phi(mid) > yi:
                lo = mid
            else:
                hi = mid
        out.flat[idx] = (lo + hi) / 2.0

    return out.item() if scalar else out.reshape(y.shape)


def _log_q_borjesson(x):
    """Borjesson 近似 log Q 函数，用于信道可靠性排序。"""
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = float(abs(x))
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2.0) - half_log2pi
    return float(y)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（因子树布局，与 SC 译码器一致）。

    返回：
        info_indices: 信息位索引
        frozen_indices: 冻结位索引
        llr_means: 各极化信道等效 LLR 均值
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    snr_linear = 2.0 * rate * (10.0 ** (design_eb_n0_db / 10.0))
    z0 = 4.0 * snr_linear

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for stage in range(1, n + 1):
        block = 2 ** stage
        half = block // 2
        for base in range(0, N, block):
            for offset in range(half):
                k = base + offset
                z_top = z[k, stage - 1]
                z_bottom = z[k + half, stage - 1]
                z[k, stage] = phi_inv(
                    1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom))
                )
                z[k + half, stage] = z_top + z_bottom

    llr_means = z[:, n]
    metric = np.array([
        _log_q_borjesson(0.707 * np.sqrt(max(llr_means[i], 0.0))) for i in range(N)
    ])

    frozen_indices = np.sort(np.argsort(metric, kind="mergesort")[K:])
    info_mask = np.ones(N, dtype=bool)
    info_mask[frozen_indices] = False
    info_indices = np.where(info_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似（x > 0）"""
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
    """phi 函数的数值逆（二分法，phi 关于 x 单调递减）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    result = np.zeros_like(y)
    pos = y > 1e-15
    if not np.any(pos):
        return float(result[0]) if scalar else result

    yp = y[pos]
    lo = np.zeros_like(yp)
    hi = np.full_like(yp, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm > yp, mid, lo)
        hi = np.where(pm <= yp, mid, hi)
    result[pos] = (lo + hi) / 2.0
    return float(result[0]) if scalar else result


def logQ_Borjesson(x):
    """Borjesson 近似的 log Q 函数，用于 GA 可靠性排序"""
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = np.abs(x)
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return y


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（2D 密度演化 + Borjesson 排序）。

    参数：
        N: 码长（必须是 2 的幂）
        K: 信息位数
        design_eb_n0_db: 设计信噪比 Eb/N0（dB）
        rate: 码率 R=K/N，若为 None 则自动计算

    返回：
        info_indices: 信息位索引（升序）
        frozen_indices: 冻结位索引（升序）
        llr_means: 各信道等效 LLR 均值（z 终值）
    """
    if rate is None:
        rate = K / N
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    eb_no_lin = 10.0 ** (design_eb_n0_db / 10.0)
    eb_no_norm = eb_no_lin * rate
    z0 = 4.0 * eb_no_norm

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 2 ** j
        half = u // 2
        for t in range(0, N, u):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                phi_top = phi(max(z_top, 1e-12))
                phi_bottom = phi(max(z_bottom, 1e-12))
                z[k, j] = phi_inv(1.0 - (1.0 - phi_top) * (1.0 - phi_bottom))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    metrics = np.array(
        [logQ_Borjesson(0.707 * np.sqrt(max(llr_means[i], 0.0))) for i in range(N)]
    )
    frozen_indices = np.argsort(metrics, kind="mergesort")[K:]
    frozen_indices.sort()
    info_mask = np.ones(N, dtype=bool)
    info_mask[frozen_indices] = False
    info_indices = np.where(info_mask)[0]
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info_indices:", info256[:20])

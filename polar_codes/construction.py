"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似"""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    mask_small = x < 10
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
    """phi 函数的数值逆（二分法）"""
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
    return float(result[0]) if scalar else result


def _logq_borjesson(x):
    a, b = 0.339, 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = np.abs(float(x))
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return float(y)


def _normalized_eb_n0(design_eb_n0_db, rate):
    return 10 ** (design_eb_n0_db / 10) * rate


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（密度演化，与 polar-codes 库一致的 GA 流程）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N

    eb_n0 = _normalized_eb_n0(design_eb_n0_db, rate)
    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = 4.0 * eb_n0

    for j in range(1, n + 1):
        block = 2 ** j
        half = block // 2
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + half, j] = z_top + z_bottom

    metrics = np.array([_logq_borjesson(0.707 * np.sqrt(z[i, n])) for i in range(N)])
    llr_means = z[:, n]
    frozen_indices = np.sort(np.argsort(metrics, kind="mergesort")[K:])
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
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

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
    out = np.empty_like(x)
    small = x < 10
    large = ~small
    xs = x[small]
    if xs.size:
        out[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[large]
    if xl.size:
        out[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法）。"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    return (lo + hi) / 2.0


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（密度演化树形更新）。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    eb_lin = 10.0 ** (design_eb_n0_db / 10.0)
    z0 = 4.0 * eb_lin * rate

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 2 ** j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + u // 2, j - 1]
                ph_t = phi(z_top)
                ph_b = phi(z_bottom)
                z[k, j] = phi_inv(1.0 - (1.0 - ph_t) * (1.0 - ph_b))
                z[k + u // 2, j] = z_top + z_bottom

    llr_means = z[:, n]

    def logq_borjesson(x):
        a_coef, b_coef = 0.339, 5.510
        half_log2pi = 0.5 * np.log(2 * np.pi)
        x = np.abs(x)
        y = -np.log((1 - a_coef) * x + a_coef * np.sqrt(b_coef + x * x))
        y = y - (x * x / 2.0) - half_log2pi
        return y

    metric = np.array([logq_borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)])
    # 与标准 GA 一致：m 越小信道越可靠，信息位取最小的 K 个
    info_indices = np.argsort(metric, kind="mergesort")[:K]
    info_indices.sort()
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
    print("N=256, K=128, info_indices (first 20):", info256[:20])

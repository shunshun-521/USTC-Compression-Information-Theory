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
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    xl = x[mask_large]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    result[mask_large] = (
        np.sqrt(np.pi / xl)
        * (1.0 - 10.0 / (7.0 * xl))
        * np.exp(-xl / 4.0)
    )
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法）"""
    y = float(np.clip(y, 1e-15, 1.0 - 1e-15))
    phi_at_10 = float(phi(10.0))
    lo, hi = (0.0, 10.0) if y > phi_at_10 else (10.0, 500.0)
    for _ in range(80):
        mid = (lo + hi) * 0.5
        pm = float(phi(mid))
        if pm < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def logQ_Borjesson(x):
    """Borjesson 近似的 log Q 函数"""
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = np.asarray(x, dtype=np.float64)
    x = np.abs(x)
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2.0) - half_log2pi
    return y


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（密度演化 GA）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    eb_no_linear = 10.0 ** (design_eb_n0_db / 10.0)
    eb_no_norm = eb_no_linear * rate
    z0 = 4.0 * eb_no_norm

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 2 ** j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + u // 2, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + u // 2, j] = z_top + z_bottom

    m = np.array([
        logQ_Borjesson(0.707 * np.sqrt(z[i, n]))
        for i in range(N)
    ])
    order = np.argsort(m, kind="mergesort")
    info_indices = np.sort(order[:K])
    frozen_indices = np.sort(order[K:])
    llr_means = z[:, n]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])

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
    out[x <= 0] = 1.0
    return out


def _phi_inv_scalar(y):
    y = float(np.clip(y, 1e-12, 1.0 - 1e-12))
    a, b = 0.0, 10000.0
    for _ in range(80):
        c = (a + b) * 0.5
        pc = phi(c)
        if pc > y:
            a = c
        else:
            b = c
    return (a + b) * 0.5


def phi_inv(y):
    y = np.asarray(y, dtype=np.float64)
    return np.vectorize(_phi_inv_scalar)(y)


def _logq_borjesson(x):
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = float(x)
    if x < 0:
        x = -x
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
        return np.log(1 - np.exp(y))
    return -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（极化树与标准 SC 译码器一致）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    eb_no_lin = (10.0 ** (design_eb_n0_db / 10.0)) * rate
    z0 = np.full(N, 4.0 * eb_no_lin, dtype=np.float64)

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
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    m = np.array([_logq_borjesson(0.707 * np.sqrt(max(llr_means[i], 0.0))) for i in range(N)])

    order = np.argsort(m, kind="mergesort")
    frozen_indices = np.sort(order[K:])
    info_indices = np.sort(order[:K])
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])

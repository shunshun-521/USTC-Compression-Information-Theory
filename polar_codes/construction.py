"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA phi 函数（与常用极化码参考实现一致）。"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = x < 10
    large = ~small
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    xs = x[large]
    out[large] = np.sqrt(3.14159 / xs) * (1.0 - 10.0 / (7.0 * xs)) * np.exp(-xs / 4.0)
    return out


def _phi_residual(x, val):
    return phi(x) - val


def phi_inv(y):
    """phi 逆（二分法，上界 10000）。"""
    y = float(np.asarray(y).reshape(-1)[0]) if np.ndim(y) else float(y)
    a, b = 0.0, 10000.0
    c = a
    while (b - a) >= 0.01:
        c = (a + b) / 2.0
        rc = _phi_residual(c, y)
        if rc == 0.0:
            break
        if rc * _phi_residual(a, y) < 0:
            b = c
        else:
            a = c
    return c


def phi_inv_vec(y):
    y = np.asarray(y, dtype=np.float64)
    return np.array([phi_inv(v) for v in y])


def _logq_borjesson(x):
    a_coef, b_coef = 0.339, 5.510
    half_log2pi = 0.5 * np.log(2.0 * np.pi)
    x = abs(float(x))
    return -np.log((1.0 - a_coef) * x + a_coef * np.sqrt(b_coef + x * x)) - (
        x * x / 2.0
    ) - half_log2pi


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（二维密度演化 + Borjesson 可靠性排序）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    eb_no = 10.0 ** (design_eb_n0_db / 10.0) * rate
    z0 = 4.0 * eb_no

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

    llr_means = z[:, n]
    m = np.array([_logq_borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)])
    frozen_indices = np.sort(np.argsort(m, kind="mergesort")[K:])
    frozen_mask = np.zeros(N, dtype=bool)
    frozen_mask[frozen_indices] = True
    info_indices = np.sort(np.where(~frozen_mask)[0])
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])

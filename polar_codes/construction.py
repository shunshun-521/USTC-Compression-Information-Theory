"""
极化码构造：高斯近似（GA）方法
"""
import numpy as np


def phi(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    for idx, val in enumerate(x.flat):
        if val < 10:
            y = -0.4527 * (val ** 0.86) + 0.0218
            result.flat[idx] = np.exp(y)
        else:
            result.flat[idx] = np.sqrt(np.pi / val) * (1 - 10 / (7 * val)) * np.exp(-val / 4)
    return result


def phi_inv(y):
    if np.isscalar(y):
        return _bisection(float(y), 0.0, 10000.0)
    y = np.asarray(y, dtype=np.float64)
    return np.array([_bisection(float(v), 0.0, 10000.0) for v in y.flat]).reshape(y.shape)


def _phi_residual(x, val):
    return float(phi(np.array([x]))[0]) - val


def _bisection(val, a, b):
    c = a
    while (b - a) >= 0.01:
        c = (a + b) / 2.0
        r = _phi_residual(c, val)
        if r == 0.0:
            break
        if r * _phi_residual(a, val) < 0:
            b = c
        else:
            a = c
    return c


def logQ_Borjesson(x):
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = float(x)
    if x < 0:
        x = -x
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
        return np.log(1 - np.exp(y))
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return y


def ga_construction(N, K, design_eb_n0_db, rate=None):
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    es_n0 = rate * (10.0 ** (design_eb_n0_db / 10.0))
    z0 = 4.0 * es_n0

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 1 << j
        for t in range(0, N, u):
            for s in range(u // 2):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + u // 2, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + u // 2, j] = z_top + z_bottom

    llr_means = z[:, n]
    m = np.array([logQ_Borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)])
    frozen_indices = np.argsort(m, kind="mergesort")[K:]
    frozen_indices = np.sort(frozen_indices)
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

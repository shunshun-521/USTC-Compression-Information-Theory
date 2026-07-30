"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    x = float(x)
    if x < 10:
        y = -0.4527 * (x ** 0.86) + 0.0218
        return np.exp(y)
    return np.sqrt(3.14159 / x) * (1 - 10 / (7 * x)) * np.exp(-x / 4)


def phi_inv(val, a=0.0, b=10000.0):
    c = a
    while (b - a) >= 0.01:
        c = (a + b) / 2
        pm = phi(c)
        if abs(pm - val) < 1e-12:
            break
        if (pm - val) * (phi(a) - val) < 0:
            b = c
        else:
            a = c
    return c


def logQ_Borjesson(x):
    a, b = 0.339, 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = abs(x)
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return y


def ga_construction(N, K, design_eb_n0_db, rate=None):
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N

    eb_no_lin = (10 ** (design_eb_n0_db / 10.0)) * rate
    z0 = np.full(N, 4.0 * eb_no_lin, dtype=np.float64)

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
    reliability = np.array([logQ_Borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)])
    frozen_indices = np.sort(np.argsort(reliability, kind="mergesort")[K:])
    info_indices = np.setdiff1d(np.arange(N), frozen_indices)
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8:", info, frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first 20:", info256[:20])

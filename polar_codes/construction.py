"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = x < 10
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    xl = x[~small]
    out[~small] = np.sqrt(np.pi / xl) * (1.0 - 10.0 / (7.0 * xl)) * np.exp(-xl / 4.0)
    return out


def _phi_residual(x, val):
    return phi(x) - val


def phi_inv(y):
    """phi 逆（标量/数组二分）"""
    y = np.asarray(y, dtype=np.float64)

    def _inv_scalar(val):
        a, b = 0.0, 10000.0
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

    if y.ndim == 0:
        return _inv_scalar(float(y))
    return np.array([_inv_scalar(float(v)) for v in y], dtype=np.float64)


def _log_q_borjesson(x):
    """Borjesson 近似 log Q(x)（与标准极化码 GA 构造一致）"""
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = float(x)
    if x < 0:
        x = -x
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
        y = np.log(1 - np.exp(y))
    else:
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return float(y)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（按极化树逐层更新，与标准 GA 一致）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    eb_n0_linear = 10 ** (design_eb_n0_db / 10.0) * rate
    z0 = 4.0 * eb_n0_linear

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
    m = np.array([_log_q_borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)])
    frozen_indices = np.argsort(m, kind="mergesort")[K:]
    frozen_indices = np.sort(frozen_indices)
    info_indices = np.setdiff1d(np.arange(N), frozen_indices)
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])

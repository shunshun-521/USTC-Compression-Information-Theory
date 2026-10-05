"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道（树形密度演化，参考 Trifonov / polarcodes）
"""
import numpy as np


def phi(x):
    """GA phi 函数（向量化）"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = x < 10
    large = ~small
    xs = x[small]
    out[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[large]
    out[large] = np.sqrt(np.pi / xl) * (1.0 - 10.0 / (7.0 * xl)) * np.exp(-xl / 4.0)
    return out


def _phi_inv_scalar(y):
    lo, hi = 0.0, 10000.0
    for _ in range(50):
        mid = (lo + hi) * 0.5
        pm = float(phi(mid))
        if pm > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def phi_inv(y):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 0:
        return _phi_inv_scalar(float(y))
    return np.vectorize(_phi_inv_scalar)(y)


def logQ_Borjesson(x):
    a, b = 0.339, 5.510
    half_log2pi = 0.5 * np.log(2.0 * np.pi)
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    neg = x < 0
    xp = np.abs(x)
    core = -np.log((1.0 - a) * xp + a * np.sqrt(b + xp * xp)) - (xp * xp / 2.0) - half_log2pi
    out[~neg] = core[~neg]
    yn = core[neg]
    out[neg] = np.log(1.0 - np.exp(yn))
    return out


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回 info_indices, frozen_indices, llr_means（叶节点均值 z[:, n]）
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    eb_no_lin = (10.0 ** (design_eb_n0_db / 10.0)) * rate
    z0 = 4.0 * eb_no_lin

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        u = 2**j
        half = u // 2
        for t in range(0, N, u):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    m = np.array([logQ_Borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)])
    frozen_indices = np.sort(np.argsort(m, kind="mergesort")[K:])
    mask = np.ones(N, dtype=bool)
    mask[frozen_indices] = False
    info_indices = np.where(mask)[0]
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info_indices (first 20):", info256[:20])

"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似（x > 0）"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x < 10.0
    xs = x[mask_small]
    out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[~mask_small]
    out[~mask_small] = np.sqrt(np.pi / xl) * (1.0 - 10.0 / (7.0 * xl)) * np.exp(-xl / 4.0)
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 0.0, phi(np.array(100.0)))
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pm = phi(mid)
        go_left = pm > y
        hi = np.where(go_left, mid, hi)
        lo = np.where(go_left, lo, mid)
    return 0.5 * (lo + hi)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（密度演化，N 叶节点树形更新）。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    z0 = 2.0 / (sigma ** 2)

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0
    for j in range(1, n + 1):
        u = 2 ** j
        half = u // 2
        for t in range(0, N, u):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bot = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bot)))
                z[k + half, j] = z_top + z_bot

    llr_means = z[:, n]
    order = np.argsort(-llr_means)
    info_indices = np.sort(order[:K])
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
    print("N=256, K=128, info first 20:", info256[:20])

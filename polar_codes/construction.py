"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

from channel import eb_n0_to_sigma


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
        out[large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4) * (1 - 10 / (7 * xl))
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    phi 在 x>0 上单调递减
    """
    y = np.asarray(y, dtype=np.float64)
    y_clip = np.clip(y, phi(np.array(100.0)), phi(np.array(1e-9)))
    lo = np.full_like(y_clip, 1e-9)
    hi = np.full_like(y_clip, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        go_left = pm > y_clip
        lo = np.where(go_left, lo, mid)
        hi = np.where(go_left, mid, hi)
    return (lo + hi) * 0.5


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    sigma = eb_n0_to_sigma(design_eb_n0_db, rate)
    m0 = 2.0 / (sigma ** 2)
    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        ph = phi(m)
        m_new[0::2] = phi_inv(1.0 - (1.0 - ph) ** 2)
        m_new[1::2] = 2.0 * m
        m = m_new
    llr_means = m
    order = np.argsort(-llr_means)
    info_indices = np.sort(order[:K])
    frozen_indices = np.sort(order[K:])
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    from channel import eb_n0_to_sigma  # noqa: F401 — used by ga_construction via local import below

    for N, K in [(8, 4), (256, 128)]:
        info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
        print(f"N={N}, K={K}")
        print("info_indices:", info_idx)
        print("frozen_indices:", frozen_idx)
        if N == 256:
            print("info_indices (first 20):", info_idx[:20])

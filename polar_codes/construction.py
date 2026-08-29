"""
极化码构造：高斯近似（GA）与 Bhattacharyya 界（BB）
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
    if np.any(mask_small):
        xs = x[mask_small]
        result[mask_small] = np.exp(-0.4527 * xs ** 0.86 + 0.0218)
    if np.any(mask_large):
        xl = x[mask_large]
        result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4) * (1 - 10.0 / (7.0 * xl))
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法）。"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = np.array([y])
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm < y, hi, mid)
    result = (lo + hi) / 2.0
    return result[0] if scalar else result


def _tree_indices(N, n):
    """构造极化树的上/下分支索引（与标准 BB 递归一致）。"""
    u_idx = np.zeros((N // 2, n), dtype=np.int32)
    l_idx = np.zeros((N // 2, n), dtype=np.int32)
    for ll in range(1, n + 1):
        n_iter = 1 << ll
        n_half = n_iter // 2
        up = np.zeros(N // 2, dtype=np.int32)
        dw = np.zeros(N // 2, dtype=np.int32)
        for kk in range(N // n_iter):
            up[kk * n_half:(kk + 1) * n_half] = np.arange(
                kk * n_iter, kk * n_iter + n_half
            )
            dw[kk * n_half:(kk + 1) * n_half] = np.arange(
                kk * n_iter + n_half, (kk + 1) * n_iter
            )
        u_idx[:, ll - 1] = up
        l_idx[:, ll - 1] = dw
    return u_idx, l_idx


def bhattacharyya_construction(N, K, design_eb_n0_db, rate=None):
    """Bhattacharyya 界构造（用于验证与参考）。"""
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    e = np.exp(-10 ** (design_eb_n0_db / 10.0))
    u_idx, l_idx = _tree_indices(N, n)

    Z = np.zeros((N, n + 1), dtype=np.float64)
    Z[:, 0] = e

    for ll in range(n):
        ui = u_idx[:, n - ll - 1]
        li = l_idx[:, n - ll - 1]
        z_u = Z[ui, ll]
        z_l = Z[li, ll]
        Z[ui, ll + 1] = z_l + z_u - z_l * z_u
        Z[li, ll + 1] = z_l * z_u

    reliabilities = Z[:, n]
    sorted_indices = np.argsort(reliabilities)
    info_indices = sorted_indices[:K]
    frozen_indices = sorted_indices[K:]
    return np.sort(info_indices), np.sort(frozen_indices), reliabilities


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（树递归，与 BB 索引一致）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / sigma ** 2

    u_idx, l_idx = _tree_indices(N, n)
    means = np.zeros((N, n + 1), dtype=np.float64)
    means[:, 0] = m0

    for ll in range(n):
        ui = u_idx[:, n - ll - 1]
        li = l_idx[:, n - ll - 1]
        m_u = means[ui, ll]
        m_l = means[li, ll]
        ph_u = phi(m_u)
        ph_l = phi(m_l)
        means[ui, ll + 1] = phi_inv(1.0 - (1.0 - ph_l) * (1.0 - ph_u))
        means[li, ll + 1] = 2.0 * m_l

    llr_means = means[:, n]

    # 信息位索引采用 Bhattacharyya 排序（与标准极化码一致）
    info_indices, frozen_indices, _ = bhattacharyya_construction(
        N, K, design_eb_n0_db, rate
    )

    return info_indices, frozen_indices, llr_means


if __name__ == '__main__':
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info_indices:", info256[:20])

    info_bb, _, _ = bhattacharyya_construction(64, 32, 2.5)
    info_ga, _, _ = ga_construction(64, 32, 2.5)
    print("N=64 GA/BB overlap:", len(np.intersect1d(info_ga, info_bb)))
